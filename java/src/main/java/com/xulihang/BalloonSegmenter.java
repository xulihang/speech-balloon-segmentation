package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.file.Paths;
import java.util.*;

/**
 * bubble-pop 气泡分割 ONNX 推理器 (漫画/多格漫画的对话气泡)
 *
 * 模型: https://github.com/letmutex/bubble-pop (MobileNetV3-Small UNet, Apache-2.0)
 *   输入:  image  [1, 1024, 768, 1]  float16 (官方 model_fp16.onnx) 或 float32
 *          灰度, letterbox 到 768x1024 (白边), (v/255 - 0.449) / 0.226
 *   输出:  logits [1, 1024, 768, 3]  与输入同精度, 依次为 mask / boundary / confidence
 *
 * 后处理 (与 bubble-pop 的 inference.py 一致):
 *   1. sigmoid, mask >= 0.5 取前景
 *   2. 8 邻域连通域, 面积 >= 60
 *   3. 实例分数 = (mask 区域内均值 + confidence 区域内均值) / 2,
 *      低于 scoreThreshold (默认 0.65) 的丢弃
 *   4. 气泡内部孔洞填充 (文字笔画), 可选内缩 shrink 像素保留轮廓线
 *   5. 可选: 传入文本框, 对掩膜覆盖不足的文本框补齐气泡
 *      (气泡被拆成几块, 或用文字定位补充模型漏掉的部分)
 *
 * 文本框格式 (原图像素):
 *   List<Map<String, Object>>, 每项 {"X", "Y", "width", "height"};
 *   也接受 {"geometry": {"X", "Y", "width", "height"}} 以及 [x0, y0, x1, y1] 数组。
 *
 * B4J 用法:
 *   Dim seg As JavaObject
 *   seg.InitializeNewInstance("com.xulihang.BalloonSegmenter", Array(modelPath))
 *   Dim mask As JavaObject = seg.RunMethod("getMask", Array(mat.JO, boxes))
 *   ' mask 是 org.opencv.core.Mat, 调用方负责 release
 */
public class BalloonSegmenter {

    private static final int HEIGHT = 1024;
    private static final int WIDTH = 768;
    private static final double MEAN = 0.449;
    private static final double STD = 0.226;
    private static final float MASK_THRESHOLD = 0.5f;
    private static final int MIN_AREA = 60;
    private static final double MAX_AREA_FRAC = 0.25;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final String outputName;
    private final boolean fp16;

    private float scoreThreshold = 0.65f;
    private int shrink = 3;
    private boolean fillHoles = true;
    private double completionCoverage = 0.9;

    public BalloonSegmenter(String onnxPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        session = env.createSession(Paths.get(onnxPath).toAbsolutePath().toString(), options);
        Map<String, NodeInfo> inputs = session.getInputInfo();
        inputName = inputs.keySet().iterator().next();
        ValueInfo info = inputs.get(inputName).getInfo();
        OnnxJavaType inputType = info instanceof TensorInfo ? ((TensorInfo) info).type : OnnxJavaType.FLOAT;
        if (inputType != OnnxJavaType.FLOAT && inputType != OnnxJavaType.FLOAT16) {
            throw new IllegalArgumentException("unsupported model input type " + inputType);
        }
        fp16 = inputType == OnnxJavaType.FLOAT16;
        Map<String, NodeInfo> outputs = session.getOutputInfo();
        outputName = outputs.keySet().iterator().next();
    }

    // 实例分数阈值, 如 (mask + confidence) / 2 >= 0.65
    public void setScoreThreshold(float threshold) {
        scoreThreshold = threshold;
    }

    // 涂白前内缩像素数 (默认 3), 0 表示不内缩
    public void setShrink(int px) {
        shrink = Math.max(0, px);
    }

    // 是否填充气泡内部孔洞 (默认 true)
    public void setFillHoles(boolean enabled) {
        fillHoles = enabled;
    }

    // 文本框覆盖率低于该比例才补洞 (默认 0.9)
    public void setCompletionCoverage(double coverage) {
        completionCoverage = Math.max(0, Math.min(1, coverage));
    }

    /**
     * 不传入文本框: 整页气泡掩膜。
     *
     * @param bgr 原图 (CV_8UC1 / CV_8UC3 / CV_8UC4, 由调用方 release)
     * @return CV_8UC1 0/255 原图尺寸掩膜, 由调用方 release
     */
    public Mat getMask(Mat bgr) {
        return getMask(bgr, null);
    }

    /**
     * 气泡掩膜, 可用文本框补洞。
     *
     * @param bgr   原图
     * @param boxes 文本框 (可为 null / 空); 见类注释的格式
     * @return CV_8UC1 0/255 原图尺寸掩膜, 由调用方 release
     */
    public Mat getMask(Mat bgr, List<?> boxes) {
        Mat mask = new Mat();
        Mat confidence = new Mat();
        try {
            predict(bgr, mask, confidence);
            Mat result = keepInstances(mask, confidence);
            try {
                if (boxes != null && !boxes.isEmpty()) {
                    Mat completed = completeWithBoxes(bgr, result, boxes);
                    result.release();
                    result = completed;
                }
                if (fillHoles) {
                    Mat filled = new Mat();
                    fillEnclosed(result, filled);
                    result.release();
                    result = filled;
                }
                if (shrink > 0) {
                    Imgproc.erode(result, result,
                            Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                                    new Size(2 * shrink + 1, 2 * shrink + 1)));
                }
                return result;
            } finally {
                mask.release();
                confidence.release();
            }
        } finally {
            mask.release();
            confidence.release();
        }
    }

    /** 只返回气泡掩膜上色的预览图 (CV_8UC3, 由调用方 release)。 */
    public Mat getOverlay(Mat bgr, List<?> boxes) {
        Mat mask = getMask(bgr, boxes);
        Mat color = bgr.channels() == 1 ? new Mat() : new Mat();
        if (bgr.channels() == 1) {
            Imgproc.cvtColor(bgr, color, Imgproc.COLOR_GRAY2BGR);
        } else {
            bgr.convertTo(color, CvType.CV_8UC3);
        }
        Mat tinted = Mat.zeros(color.size(), CvType.CV_8UC3);
        Mat green = new Mat(color.size(), CvType.CV_8UC3, new Scalar(0, 200, 0));
        green.copyTo(tinted, mask);
        Mat out = new Mat();
        Core.addWeighted(tinted, 0.6, color, 0.4, 0, out);
        mask.release();
        color.release();
        tinted.release();
        green.release();
        return out;
    }

    public void close() {
        try {
            session.close();
            env.close();
        } catch (OrtException e) {
            // already closed
        }
    }

    // ------------------------------------------------------------------ inference

    private void predict(Mat image, Mat maskOut, Mat confidenceOut) {
        int srcW = image.cols();
        int srcH = image.rows();
        Mat gray = new Mat();
        if (image.channels() == 1) {
            image.copyTo(gray);
        } else if (image.channels() == 4) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGRA2GRAY);
        } else {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        }

        double scale = Math.min((double) WIDTH / srcW, (double) HEIGHT / srcH);
        int rw = Math.max(1, (int) Math.round(srcW * scale));
        int rh = Math.max(1, (int) Math.round(srcH * scale));
        int padX = (WIDTH - rw) / 2;
        int padY = (HEIGHT - rh) / 2;

        Mat resized = new Mat();
        Imgproc.resize(gray, resized, new Size(rw, rh), 0, 0,
                scale < 1 ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
        gray.release();
        byte[] pixels = new byte[rh * rw];
        resized.get(0, 0, pixels);
        resized.release();

        int n = HEIGHT * WIDTH;
        float[] values = new float[n];
        Arrays.fill(values, (float) ((255.0 / 255.0 - MEAN) / STD));
        for (int y = 0; y < rh; y++) {
            int rowOffset = y * rw;
            int outOffset = (y + padY) * WIDTH + padX;
            for (int x = 0; x < rw; x++) {
                int v = pixels[rowOffset + x] & 0xFF;
                values[outOffset + x] = (float) ((v / 255.0 - MEAN) / STD);
            }
        }

        long[] shape = {1, HEIGHT, WIDTH, 1};
        float[] logits;
        try (OnnxTensor input = createInput(values, shape);
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, input))) {
            OnnxValue value = result.get(outputName)
                    .orElseThrow(() -> new IllegalStateException("missing output " + outputName));
            if (fp16) {
                ShortBuffer buffer = ((OnnxTensor) value).getShortBuffer();
                logits = new float[buffer.remaining()];
                for (int i = 0; i < logits.length; i++) {
                    logits[i] = halfToFloat(buffer.get(i));
                }
            } else {
                FloatBuffer buffer = ((OnnxTensor) value).getFloatBuffer();
                logits = new float[buffer.remaining()];
                buffer.get(logits);
            }
        } catch (OrtException e) {
            throw new RuntimeException("inference failed: " + e.getMessage(), e);
        }
        // [1, HEIGHT, WIDTH, 3] 摊平后, 第 (y*WIDTH + x)*3 + c 个值是 (x, y) 的通道 c
        final int channels = 3;
        Mat plane = new Mat(rh, rw, CvType.CV_32FC1);
        float[] planeData = new float[rh * rw];
        plane.put(0, 0, planeData);
        for (int c = 0; c < channels; c++) {
            int k = 0;
            for (int y = 0; y < rh; y++) {
                int base = ((y + padY) * WIDTH + padX) * channels + c;
                for (int x = 0; x < rw; x++) {
                    planeData[k++] = logits[base + x * channels];
                }
            }
            plane.put(0, 0, planeData);
            Mat full = new Mat();
            Imgproc.resize(plane, full, new Size(srcW, srcH), 0, 0, Imgproc.INTER_LINEAR);
            sigmoid(full);
            if (c == 0) {
                full.copyTo(maskOut);
            } else if (c == 2) {
                full.copyTo(confidenceOut);
            }
            full.release();
        }
        plane.release();
    }

    /** 输入张量: 模型是 float16 时把数值转成半精度 short。 */
    private OnnxTensor createInput(float[] values, long[] shape) throws OrtException {
        if (!fp16) {
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(values), shape);
        }
        short[] half = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            half[i] = toHalf(values[i]);
        }
        return OnnxTensor.createTensor(env, ShortBuffer.wrap(half), shape, OnnxJavaType.FLOAT16);
    }

    /** IEEE 754 binary16 转换 (Java 8 没有 Float.floatToFloat16)。 */
    static short toHalf(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exp = (bits >>> 23) & 0xFF;
        int mant = bits & 0x7FFFFF;
        if (exp == 255) {
            return (short) (sign | 0x7C00 | (mant >>> 13));
        }
        int e = exp - 127 + 15;
        if (e >= 31) {
            return (short) (sign | 0x7C00);
        }
        if (e <= 0) {
            if (e < -10) {
                return (short) sign;
            }
            mant |= 0x800000;
            int shift = 14 - e;
            int rounded = (mant + (1 << (shift - 1))) >>> shift;
            return (short) (sign | rounded);
        }
        return (short) (sign | (e << 10) | (mant >>> 13));
    }

    static float halfToFloat(short h) {
        int bits = h & 0xFFFF;
        int sign = (bits >>> 15) & 1;
        int exp = (bits >>> 10) & 0x1F;
        int mant = bits & 0x3FF;
        float f;
        if (exp == 0) {
            f = (float) (mant * Math.pow(2, -24));
        } else if (exp == 31) {
            f = Float.NaN;
        } else {
            f = (float) ((1.0 + mant / 1024.0) * Math.pow(2, exp - 15));
        }
        return sign == 1 ? -f : f;
    }

    private static void sigmoid(Mat mat) {
        int rows = mat.rows();
        int cols = mat.cols();
        float[] row = new float[cols];
        for (int y = 0; y < rows; y++) {
            mat.get(y, 0, row);
            for (int x = 0; x < cols; x++) {
                double v = Math.max(-88.0, Math.min(88.0, row[x]));
                row[x] = (float) (1.0 / (1.0 + Math.exp(-v)));
            }
            mat.put(y, 0, row);
        }
    }

    // ------------------------------------------------------------------ post-processing

    private Mat keepInstances(Mat mask, Mat confidence) {
        Mat binary = new Mat();
        Imgproc.threshold(mask, binary, MASK_THRESHOLD, 255, Imgproc.THRESH_BINARY);
        if (binary.depth() != CvType.CV_8U) {
            // threshold keeps the float type of the input
            Mat binary8 = new Mat();
            binary.convertTo(binary8, CvType.CV_8U);
            binary.release();
            binary = binary8;
        }
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int count = Imgproc.connectedComponentsWithStats(binary, labels, stats, centroids, 8, CvType.CV_32S);
        binary.release();
        stats.release();
        centroids.release();

        int total = mask.rows() * mask.cols();
        int[] labelData = new int[total];
        labels.get(0, 0, labelData);
        labels.release();
        float[] maskData = new float[total];
        mask.get(0, 0, maskData);
        float[] confData = new float[total];
        confidence.get(0, 0, confData);

        double[] sumMask = new double[count];
        double[] sumConf = new double[count];
        int[] area = new int[count];
        for (int i = 0; i < total; i++) {
            int l = labelData[i];
            if (l > 0) {
                sumMask[l] += maskData[i];
                sumConf[l] += confData[i];
                area[l]++;
            }
        }
        boolean[] keep = new boolean[count];
        for (int l = 1; l < count; l++) {
            if (area[l] < MIN_AREA) {
                continue;
            }
            double score = (sumMask[l] / area[l] + sumConf[l] / area[l]) / 2;
            keep[l] = score >= scoreThreshold;
        }
        byte[] outData = new byte[total];
        for (int i = 0; i < total; i++) {
            if (labelData[i] > 0 && keep[labelData[i]]) {
                outData[i] = (byte) 255;
            }
        }
        Mat out = new Mat(mask.rows(), mask.cols(), CvType.CV_8UC1);
        out.put(0, 0, outData);
        return out;
    }

    /** 填充被掩膜围住的区域 (气泡内的文字笔画)。 */
    private static void fillEnclosed(Mat mask, Mat out) {
        Mat padded = new Mat(mask.rows() + 2, mask.cols() + 2, CvType.CV_8UC1, new Scalar(0));
        Mat dst = padded.submat(new Rect(1, 1, mask.cols(), mask.rows()));
        mask.copyTo(dst);
        Mat flood = padded.clone();
        Imgproc.floodFill(flood, new Mat(), new Point(0, 0), new Scalar(255));
        Mat inverted = new Mat();
        Core.bitwise_not(flood, inverted);
        Mat holes = inverted.submat(new Rect(1, 1, mask.cols(), mask.rows()));
        Core.bitwise_or(mask, holes, out);
        padded.release();
        flood.release();
        inverted.release();
    }

    // ------------------------------------------------------------------ text completion

    private Mat completeWithBoxes(Mat image, Mat current, List<?> boxes) {
        Mat gray = new Mat();
        if (image.channels() == 1) {
            image.copyTo(gray);
        } else if (image.channels() == 4) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGRA2GRAY);
        } else {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        }
        Mat out = current.clone();
        for (Object box : boxes) {
            int[] r = parseBox(box);
            if (r != null) {
                completeOne(gray, out, r);
            }
        }
        gray.release();
        return out;
    }

    /** 文本框 -> [x0, y0, x1, y1] (右下开区间), 支持 Map / Ma / 数组。 */
    static int[] parseBox(Object box) {
        int x0, y0, x1, y1;
        if (box instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) box;
            Object geometry = m.get("geometry");
            if (geometry instanceof Map) {
                m = (Map<?, ?>) geometry;
            }
            if (!m.containsKey("X")) {
                return null;
            }
            x0 = toInt(m.get("X"));
            y0 = toInt(m.get("Y"));
            x1 = x0 + toInt(m.get("width"));
            y1 = y0 + toInt(m.get("height"));
        } else if (box instanceof int[]) {
            int[] b = (int[]) box;
            if (b.length != 4) {
                return null;
            }
            x0 = b[0];
            y0 = b[1];
            x1 = b[2];
            y1 = b[3];
        } else if (box instanceof Number[]) {
            Number[] b = (Number[]) box;
            if (b.length != 4) {
                return null;
            }
            x0 = b[0].intValue();
            y0 = b[1].intValue();
            x1 = b[2].intValue();
            y1 = b[3].intValue();
        } else if (box instanceof float[]) {
            float[] b = (float[]) box;
            if (b.length != 4) {
                return null;
            }
            x0 = Math.round(b[0]);
            y0 = Math.round(b[1]);
            x1 = Math.round(b[2]);
            y1 = Math.round(b[3]);
        } else if (box instanceof List) {
            List<?> b = (List<?>) box;
            if (b.size() != 4) {
                return null;
            }
            x0 = toInt(b.get(0));
            y0 = toInt(b.get(1));
            x1 = toInt(b.get(2));
            y1 = toInt(b.get(3));
        } else {
            return null;
        }
        return new int[]{x0, y0, x1, y1};
    }

    private static int toInt(Object value) {
        return value instanceof Number ? ((Number) value).intValue() : Math.round(Float.parseFloat(value.toString()));
    }

    private void completeOne(Mat gray, Mat out, int[] box) {
        int x0 = Math.max(0, box[0]);
        int y0 = Math.max(0, box[1]);
        int x1 = Math.min(out.cols() - 1, box[2]);
        int y1 = Math.min(out.rows() - 1, box[3]);
        if (x1 <= x0 || y1 <= y0) {
            return;
        }
        int w = x1 - x0 + 1;
        int h = y1 - y0 + 1;

        Mat roi = out.submat(new Rect(x0, y0, w, h));
        double covered = Core.countNonZero(roi) / (double) (w * h);
        if (covered >= completionCoverage) {
            return;
        }

        int margin = 80;
        int wx0 = Math.max(0, x0 - margin);
        int wy0 = Math.max(0, y0 - margin);
        int ww = Math.min(gray.cols(), wx0 + w + 2 * margin) - wx0;
        int wh = Math.min(gray.rows(), wy0 + h + 2 * margin) - wy0;
        Mat win = gray.submat(new Rect(wx0, wy0, ww, wh));
        Mat ink = new Mat();
        Imgproc.threshold(win, ink, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        // 文本自身的字形不作为墙 (否则气泡内部被字形切碎)
        ink.submat(new Rect(x0 - wx0, y0 - wy0, w, h)).setTo(new Scalar(0));
        Mat white = new Mat();
        Imgproc.threshold(win, white, 128, 255, Imgproc.THRESH_BINARY);

        // 内轮廓的缺口很小, 用逐级变薄的墙让轮廓闭合
        for (int gap = 3; gap >= 1; gap -= 2) {
            Mat wall = new Mat();
            Imgproc.dilate(ink, wall,
                    Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2 * gap + 1, 2 * gap + 1)));
            Mat free = new Mat();
            Core.bitwise_not(wall, free);
            wall.release();
            Mat freeWhite = new Mat();
            Core.bitwise_and(free, white, freeWhite);
            free.release();
            Mat labels = new Mat();
            Mat stats = new Mat();
            Mat centroids = new Mat();
            int num = Imgproc.connectedComponentsWithStats(freeWhite, labels, stats, centroids, 8, CvType.CV_32S);
            freeWhite.release();
            stats.release();
            centroids.release();
            if (num <= 1) {
                labels.release();
                continue;
            }
            int[] labelData = new int[ww * wh];
            labels.get(0, 0, labelData);
            labels.release();
            // 文本框中间一行出现最多的标签
            int midRow = (y0 - wy0 + h / 2) * ww + x0 - wx0;
            Map<Integer, Integer> counts = new HashMap<>();
            for (int i = 0; i < w; i++) {
                int l = labelData[midRow + i];
                if (l > 0) {
                    counts.merge(l, 1, Integer::sum);
                }
            }
            int best = 0;
            int bestCount = 0;
            for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
                if (e.getValue() > bestCount) {
                    best = e.getKey();
                    bestCount = e.getValue();
                }
            }
            if (best == 0) {
                continue;
            }
            byte[] compData = new byte[ww * wh];
            for (int i = 0; i < labelData.length; i++) {
                compData[i] = labelData[i] == best ? (byte) 255 : 0;
            }
            Mat comp = new Mat(wh, ww, CvType.CV_8UC1);
            comp.put(0, 0, compData);
            comp.submat(new Rect(x0 - wx0, y0 - wy0, w, h)).setTo(new Scalar(255));
            // 只在它与模型已找到的气泡相接时才补 (同一气泡可能被拆成几块)
            Mat found = new Mat(wh, ww, CvType.CV_8UC1, new Scalar(0));
            out.submat(new Rect(wx0, wy0, ww, wh)).copyTo(found);
            Mat near = new Mat();
            Imgproc.dilate(found, near,
                    Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(9, 9)));
            found.release();
            Mat touch = new Mat();
            Core.bitwise_and(comp, near, touch);
            near.release();
            if (Core.countNonZero(touch) > 0) {
                Mat pageComp = Mat.zeros(out.size(), CvType.CV_8UC1);
                comp.copyTo(pageComp.submat(new Rect(wx0, wy0, ww, wh)));
                Core.bitwise_or(out, pageComp, out);
                pageComp.release();
                System.out.println("balloon seg: text box " + x0 + "," + y0
                        + " completed (" + Math.round(covered * 100) + "% covered)");
            }
            touch.release();
            comp.release();
            break;
        }
        win.release();
        ink.release();
        white.release();
    }

    // ------------------------------------------------------------------ main

    /** 命令行: BalloonSegmenter <model.onnx> <image> <out.png> [x0,y0,x1,y1; ...] */
    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        if (args.length < 3) {
            System.out.println("usage: BalloonSegmenter <model.onnx> <image> <out.png> [x0,y0,x1,y1;...]");
            return;
        }
        BalloonSegmenter segmenter = new BalloonSegmenter(args[0]);
        Mat image = org.opencv.imgcodecs.Imgcodecs.imread(args[1]);
        if (image.empty()) {
            System.err.println("cannot read " + args[1]);
            return;
        }
        List<Object> boxes = new ArrayList<>();
        if (args.length > 3 && !args[3].isEmpty()) {
            for (String part : args[3].split(";")) {
                String[] v = part.trim().split(",");
                boxes.add(new int[]{
                        Integer.parseInt(v[0].trim()), Integer.parseInt(v[1].trim()),
                        Integer.parseInt(v[2].trim()), Integer.parseInt(v[3].trim())});
            }
        }
        long start = System.nanoTime();
        Mat mask = segmenter.getMask(image, boxes);
        long ms = (System.nanoTime() - start) / 1_000_000;
        int px = Core.countNonZero(mask);
        org.opencv.imgcodecs.Imgcodecs.imwrite(args[2], mask);
        System.out.printf("%dx%d, mask %d px, %.0f ms -> %s%n",
                image.cols(), image.rows(), px, (double) ms, args[2]);
        mask.release();
        image.release();
        segmenter.close();
    }
}
