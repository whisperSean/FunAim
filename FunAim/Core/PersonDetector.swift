import CoreML
import CoreVideo
import Foundation

/// 单个检测结果
/// box 为归一化坐标（原点左下）；label 为类别名
struct DetectedPerson {
    let id: UUID
    let box: CGRect
    let confidence: Float
    let label: String
}

/// 人物检测器：直读 YOLO11n CoreML 原始输出张量
/// 不经过 Vision 的 NMS 管线/标签解释（该路径类别映射不可靠），
/// 在张量层面只取 class 0（COCO 中 person 恒为 0 号类）——
/// 非人物类别在解码层即被排除，不存在误画可能。
/// 本地完成置信度过滤 + 越界收敛 + IoU NMS，输出归一化框。
final class PersonDetector {

    static let coreMLModelName = "yolo11n"

    /// COCO 80 类标签（class 0 = person）
    private static let cocoLabels: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    private static let personClassIndex = 0
    /// person 类硬性置信度下限：低置信度误检直接丢弃
    private static let personFloor: Float = 0.6
    private static let inputSize: Float = 640

    private var model: MLModel?
    private var inputName = "image"
    private var outputName: String?
    private let queue = DispatchQueue(label: "funaim.inference", qos: .userInteractive)
    private var tracker = PersonTracker()

    init() {
        guard let url = Self.locateCoreMLModel() else { return }
        let config = MLModelConfiguration()
        config.computeUnits = .all   // ANE / GPU / CPU 自动调度
        guard let loaded = try? MLModel(contentsOf: url, configuration: config) else { return }
        model = loaded
        inputName = loaded.modelDescription.inputDescriptionsByName.first?.key ?? "image"
        outputName = loaded.modelDescription.outputDescriptionsByName
            .first { $0.value.type == .multiArray }?.key
    }

    /// 推理预热：消除首次 prediction 的冷启动开销（会话启动时调用一次）
    func warmup() {
        queue.async { [weak self] in
            guard let self, let model = self.model else { return }
            var pb: CVPixelBuffer?
            let attrs: [CFString: Any] = [kCVPixelBufferWidthKey: Int(Self.inputSize),
                                          kCVPixelBufferHeightKey: Int(Self.inputSize)]
            CVPixelBufferCreate(kCFAllocatorDefault, Int(Self.inputSize), Int(Self.inputSize),
                                kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
            guard let buffer = pb else { return }
            if let input = try? Self.featureProvider(buffer, inputName: self.inputName) {
                _ = try? model.prediction(from: input)
            }
        }
    }

    /// 异步检测，completion 回调主线程
    func detect(pixelBuffer: CVPixelBuffer,
                minConfidence: Float,
                completion: @escaping ([DetectedPerson], Double) -> Void) {
        guard let model, let outputName else {
            DispatchQueue.main.async { completion([], 0) }
            return
        }
        queue.async { [weak self] in
            guard let self else { return }
            let t0 = CFAbsoluteTimeGetCurrent()
            do {
                let input = try Self.featureProvider(pixelBuffer, inputName: self.inputName)
                let output = try model.prediction(from: input)
                guard let multiArray = output.featureValue(for: outputName)?.multiArrayValue else {
                    DispatchQueue.main.async { completion([], 0) }
                    return
                }
                let costMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

                // 张量解码 + 类别过滤（每帧直读设置中心，切换即时生效）
                let showAll = AppSettings.shared.showAllClasses
                let raw = Self.decode(multiArray, showAll: showAll, minConfidence: minConfidence)
                let filtered = Self.nms(raw, iouThreshold: 0.45)

                let persons = self.tracker.assign(filtered)
                DispatchQueue.main.async { completion(persons, costMs) }
            } catch {
                DispatchQueue.main.async { completion([], 0) }
            }
        }
    }

    // MARK: - 原始张量解码

    /// YOLO 输出 [1, 84, 8400]（4 框 + 80 类）或转置 [1, 8400, 84]
    /// 每个候选：cx, cy, w, h（模型输入像素坐标）+ 80 个类别分数
    private static func decode(_ mla: MLMultiArray, showAll: Bool, minConfidence: Float) -> [(box: CGRect, conf: Float, label: String)] {
        let shape = mla.shape.map { $0.intValue }
        guard shape.count == 3 else { return [] }

        let channels: Int
        let count: Int
        let channelStride: Int
        let indexStride: Int
        if shape[1] <= shape[2] {
            channels = shape[1]; count = shape[2]
            channelStride = mla.strides[1].intValue; indexStride = mla.strides[2].intValue
        } else {
            channels = shape[2]; count = shape[1]
            channelStride = mla.strides[2].intValue; indexStride = mla.strides[1].intValue
        }
        let labelCount = channels - 4
        guard labelCount > personClassIndex else { return [] }

        var result: [(box: CGRect, conf: Float, label: String)] = []
        for i in 0..<count {
            let base = i * indexStride

            if !showAll {
                // 仅 person：只读一个通道，快速跳过非人物候选
                let score = mla[base + personClassIndex * channelStride + 4 * channelStride].floatValue
                guard score >= max(minConfidence, personFloor) else { continue }
                let cx = mla[base].doubleValue
                let cy = mla[base + channelStride].doubleValue
                let w = mla[base + 2 * channelStride].doubleValue
                let h = mla[base + 3 * channelStride].doubleValue
                appendBox(cx, cy, w, h, score, personClassIndex, &result)
                continue
            }

            // 全类别：找最高分类别
            var bestClass = -1
            var bestScore = Float(0)
            for c in 0..<labelCount {
                let s = mla[base + (4 + c) * channelStride].floatValue
                if s > bestScore { bestScore = s; bestClass = c }
            }
            guard bestClass >= 0 else { continue }
            let isPerson = bestClass == personClassIndex
            let threshold: Float = isPerson ? max(minConfidence, personFloor) : minConfidence
            guard bestScore >= threshold else { continue }
            let cx = mla[base].doubleValue
            let cy = mla[base + channelStride].doubleValue
            let w = mla[base + 2 * channelStride].doubleValue
            let h = mla[base + 3 * channelStride].doubleValue
            appendBox(cx, cy, w, h, bestScore, bestClass, &result)
        }
        return result
    }

    private static func appendBox(_ cx: Double, _ cy: Double, _ w: Double, _ h: Double,
                                  _ score: Float, _ classIndex: Int,
                                  _ result: inout [(box: CGRect, conf: Float, label: String)]) {
        let box = CGRect(x: (cx - w / 2) / Double(inputSize),
                         y: (cy - h / 2) / Double(inputSize),
                         width: w / Double(inputSize),
                         height: h / Double(inputSize))
        result.append((box: box, conf: score, label: cocoLabels[classIndex]))
    }

    // MARK: - 输入构造 / 模型加载

    private static func featureProvider(_ pixelBuffer: CVPixelBuffer, inputName: String) throws -> MLFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)])
    }

    private static func locateCoreMLModel() -> URL? {
        let name = coreMLModelName
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return url
        }
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodel") {
            return try? MLModel.compileModel(at: url)   // 一次性编译到临时目录
        }
        if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            return try? MLModel.compileModel(at: url)   // 新版 coremltools 输出格式
        }
        return nil
    }

    /// 越界/过小框：收敛到画面内并剔除无效框
    private static func clamp(_ detection: (box: CGRect, conf: Float, label: String)) -> (box: CGRect, conf: Float, label: String)? {
        let clamped = detection.box.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard !clamped.isNull, clamped.width > 0.03, clamped.height > 0.03 else { return nil }
        return (box: clamped, conf: detection.conf, label: detection.label)
    }

    /// IoU 非极大值抑制
    private static func nms(_ input: [(box: CGRect, conf: Float, label: String)], iouThreshold: Float) -> [(box: CGRect, conf: Float, label: String)] {
        let sorted = input.sorted { $0.conf > $1.conf }
        var keep: [(box: CGRect, conf: Float, label: String)] = []
        for candidate in sorted where keep.allSatisfy({ iou($0.box, candidate.box) < iouThreshold }) {
            keep.append(candidate)
        }
        return keep.compactMap(clamp)
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        guard unionArea > 0 else { return 0 }
        return Float(interArea / unionArea)
    }
}

/// 轻量跟踪：IoU 优先 + 中心距离兜底的双级匹配，
/// 检测框 EMA 平滑 + 线性速度外推 + 短暂遮挡滑行，保证目标 ID 跨帧稳定、锁定不跳人
final class PersonTracker {

    private struct Track {
        let id: UUID
        var rect: CGRect
        var velocity: CGSize
        var confidence: Float
        var label: String
        var misses: Int
    }

    private var tracks: [Track] = []
    /// 平滑系数：越大越贴测量值（0.6 ≈ 抑制抖动且延迟极小）
    private let smoothAlpha: CGFloat = 0.6
    /// 中心距离兜底匹配阈值（归一化坐标）
    private let matchThreshold: CGFloat = 0.14
    /// IoU 匹配阈值
    private let iouMatchThreshold: CGFloat = 0.25
    /// 遮挡滑行帧数（推理帧）：仅保留 1 帧，尽量不渲染"幽灵框"
    private let maxCoastMisses = 1

    func assign(_ detections: [(box: CGRect, conf: Float, label: String)]) -> [DetectedPerson] {
        // 1) 线性速度外推：补偿推理间隔内的位移
        for index in tracks.indices {
            tracks[index].rect = tracks[index].rect.offsetBy(dx: tracks[index].velocity.width,
                                                             dy: tracks[index].velocity.height)
        }

        var detectionUsed = [Bool](repeating: false, count: detections.count)
        var trackMatched = [Bool](repeating: false, count: tracks.count)
        var result: [DetectedPerson] = []
        result.reserveCapacity(detections.count + tracks.count)

        // 2) 第一优先：IoU 匹配（位置+大小同时吻合，抗交叉换 ID）
        for (dIndex, detection) in detections.enumerated() {
            var bestIndex = -1
            var bestIoU = iouMatchThreshold
            for (tIndex, track) in tracks.enumerated() where !trackMatched[tIndex] {
                let score = Self.iou(track.rect, detection.box)
                if score > bestIoU { bestIoU = score; bestIndex = tIndex }
            }
            if bestIndex >= 0 {
                detectionUsed[dIndex] = true
                trackMatched[bestIndex] = true
                result.append(merge(index: bestIndex, detection))
            }
        }

        // 3) 第二优先：中心距离兜底（新入画/大位移目标）
        for (dIndex, detection) in detections.enumerated() where !detectionUsed[dIndex] {
            let center = CGPoint(x: detection.box.midX, y: detection.box.midY)
            var bestIndex = -1
            var bestDistance = matchThreshold
            for (tIndex, track) in tracks.enumerated() where !trackMatched[tIndex] {
                let distance = hypot(track.rect.midX - center.x, track.rect.midY - center.y)
                if distance < bestDistance { bestDistance = distance; bestIndex = tIndex }
            }
            if bestIndex >= 0 {
                trackMatched[bestIndex] = true
                result.append(merge(index: bestIndex, detection))
            } else {
                result.append(DetectedPerson(id: UUID(), box: detection.box,
                                             confidence: detection.conf, label: detection.label))
            }
        }

        // 4) 未匹配轨迹：按速度滑行输出（置信度衰减），超限移除
        for (tIndex, track) in tracks.enumerated() where !trackMatched[tIndex] {
            var coasted = track
            coasted.misses += 1
            if coasted.misses <= maxCoastMisses {
                coasted.confidence *= 0.8
                tracks[tIndex] = coasted
                result.append(DetectedPerson(id: coasted.id, box: coasted.rect,
                                             confidence: coasted.confidence, label: coasted.label))
            }
        }

        // 5) 用本帧结果重建轨迹表（merge/滑行均已把最新值写回）
        tracks = result.map { person in
            if let index = tracks.firstIndex(where: { $0.id == person.id }) {
                return tracks[index]
            }
            return Track(id: person.id, rect: person.box, velocity: .zero,
                         confidence: person.confidence, label: person.label, misses: 0)
        }
        return result
    }

    /// EMA 更新 + 速度估计
    private func merge(index: Int, _ detection: (box: CGRect, conf: Float, label: String)) -> DetectedPerson {
        var track = tracks[index]
        let previousCenter = CGPoint(x: track.rect.midX, y: track.rect.midY)
        let box = CGRect(
            x: track.rect.minX + (detection.box.minX - track.rect.minX) * smoothAlpha,
            y: track.rect.minY + (detection.box.minY - track.rect.minY) * smoothAlpha,
            width: track.rect.width + (detection.box.width - track.rect.width) * smoothAlpha,
            height: track.rect.height + (detection.box.height - track.rect.height) * smoothAlpha
        )
        let newCenter = CGPoint(x: box.midX, y: box.midY)
        let residual = CGSize(width: newCenter.x - previousCenter.x,
                              height: newCenter.y - previousCenter.y)
        track.velocity = CGSize(width: track.velocity.width * 0.5 + residual.width * 0.5,
                                height: track.velocity.height * 0.5 + residual.height * 0.5)
        track.rect = box
        track.confidence = detection.conf
        track.label = detection.label
        track.misses = 0
        tracks[index] = track
        return DetectedPerson(id: track.id, box: box, confidence: detection.conf, label: detection.label)
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        guard unionArea > 0 else { return 0 }
        return interArea / unionArea
    }
}
