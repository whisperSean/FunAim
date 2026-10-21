import CoreImage
import CoreML
import CoreVideo
import Foundation

/// Letterbox 变换参数：源画面等比缩放后居中放入 640×640 输入，四周黑边填充
/// 解码时用这组参数把模型坐标还原回源画面归一化坐标
struct Letterbox {
    /// 内容在模型输入中的像素宽/高（≤640）
    let contentW: Double
    let contentH: Double
    let padX: Double
    let padY: Double

    /// 单位变换（仅用于单元测试）
    static let identity = Letterbox(contentW: 640, contentH: 640, padX: 0, padY: 0)
}

/// 解码中间结构（跟踪器输入）
struct RawTarget {
    let box: CGRect
    let conf: Float
    let label: String
    let isPerson: Bool
    let state: PersonState
    let faceVisible: Bool
    let estimatedHeightCm: Double?
}

/// 双模型检测器：
/// - YOLO11n（检测）：生活物品白名单 32 类（可在设置关闭）
/// - YOLO11n-pose（姿态）：人物 + 17 关键点 → 站立/蹲姿/半身 + 人脸露出
/// 两者均为直读原始张量解码（无 NMS 管线/无 Vision 标签解释），class 索引硬过滤；
/// 每帧各推理一次（ANE 调度），本地完成置信度过滤 + 越界收敛 + IoU NMS。
final class PersonDetector {

    static let detModelName = "yolo11n"
    static let poseModelName = "yolo11n-pose"

    /// COCO 80 类标签（class 0 = person），物品解码按下标取类别名
    static let cocoLabels: [String] = [
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

    /// 生活物品白名单
    static let householdItems: Set<String> = [
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "book", "scissors", "teddy bear",
        "hair drier", "toothbrush", "chair", "couch", "potted plant", "bed",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "sink", "refrigerator", "clock", "vase"
    ]

    private static let personFloor: Float = 0.6
    private static let itemFloor: Float = 0.5
    private static let inputSize: Float = 640

    private var detModel: MLModel?
    private var poseModel: MLModel?
    private var detInputName = "image"
    private var detOutputName: String?
    private var poseInputName = "image"
    private var poseOutputName: String?
    private let queue = DispatchQueue(label: "funaim.inference", qos: .userInteractive)
    private var tracker = PersonTracker()

    /// 运行时诊断回调（主线程）：模型加载状态 / 各模型检出数 / 推理错误
    var onDiagnostic: ((String) -> Void)?
    private var loadStatus = "初始化中"

    init() {
        detModel = Self.loadModel(Self.detModelName)
        loadStatus = detModel == nil ? "det✗加载失败" : "det✓"
        if let m = detModel {
            detInputName = m.modelDescription.inputDescriptionsByName.first?.key ?? "image"
            detOutputName = m.modelDescription.outputDescriptionsByName
                .first { $0.value.type == .multiArray }?.key
        }
        if detModel != nil, detOutputName == nil { loadStatus += "(输出缺失)" }

        poseModel = Self.loadModel(Self.poseModelName)
        loadStatus += poseModel == nil ? " | pose✗加载失败" : " | pose✓"
        if let m = poseModel {
            poseInputName = m.modelDescription.inputDescriptionsByName.first?.key ?? "image"
            poseOutputName = m.modelDescription.outputDescriptionsByName
                .first { $0.value.type == .multiArray }?.key
        }
        if poseModel != nil, poseOutputName == nil { loadStatus += "(输出缺失)" }
    }

    private func report(_ detCount: Int, _ poseCount: Int, _ error: String?) {
        var text = loadStatus + " | det:\(detCount) pose:\(poseCount)"
        if let error { text += " | ⚠️\(error)" }
        DispatchQueue.main.async { [weak self] in
            self?.onDiagnostic?(text)
        }
    }

    /// 双模型推理预热
    func warmup() {
        queue.async { [weak self] in
            guard let self else { return }
            var pb: CVPixelBuffer?
            let attrs: [CFString: Any] = [kCVPixelBufferWidthKey: Int(Self.inputSize),
                                          kCVPixelBufferHeightKey: Int(Self.inputSize)]
            CVPixelBufferCreate(kCFAllocatorDefault, Int(Self.inputSize), Int(Self.inputSize),
                                kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
            guard let buffer = pb else { return }
            if let detModel = self.detModel,
               let input = try? Self.featureProvider(buffer, inputName: self.detInputName) {
                _ = try? detModel.prediction(from: input)
            }
            if let poseModel = self.poseModel,
               let input = try? Self.featureProvider(buffer, inputName: self.poseInputName) {
                _ = try? poseModel.prediction(from: input)
            }
        }
    }

    /// 异步双模型检测，completion 回调主线程
    func detect(pixelBuffer: CVPixelBuffer,
                minConfidence: Float,
                completion: @escaping ([DetectedPerson], Double) -> Void) {
        queue.async { [weak self] in
            guard let self else { return }
            let t0 = CFAbsoluteTimeGetCurrent()

            let settings = AppSettings.shared
            var rawTargets: [RawTarget] = []
            var detCount = 0
            var poseCount = 0
            var inferenceError: String?

            // 0) 预处理：模型只接受 640×640 输入，摄像头原始帧必须 letterbox 后再送推理
            guard let (modelInput, letterbox) = Self.makeLetterboxedBuffer(pixelBuffer) else {
                self.report(0, 0, "预处理失败")
                DispatchQueue.main.async { completion([], 0) }
                return
            }

            // 1) 姿态模型：人物框 + 17 关键点（人物唯一来源，避免双模型重复框）
            if settings.poseEnabled, let poseModel = self.poseModel, let outName = self.poseOutputName {
                do {
                    let input = try Self.featureProvider(modelInput, inputName: self.poseInputName)
                    let output = try poseModel.prediction(from: input)
                    if let mla = output.featureValue(for: outName)?.multiArrayValue {
                        let decoded = Self.decodePose(mla, minConfidence: minConfidence, letterbox: letterbox)
                        poseCount = decoded.count
                        rawTargets += decoded
                    } else {
                        inferenceError = "pose输出类型异常"
                    }
                } catch {
                    inferenceError = "pose:\(error.localizedDescription)"
                }
            }

            // 2) 检测模型：生活物品白名单；姿态关闭时 person 也由此模型给出（无姿态信息）
            if settings.itemsEnabled || !settings.poseEnabled,
               let detModel = self.detModel, let outName = self.detOutputName {
                do {
                    let input = try Self.featureProvider(modelInput, inputName: self.detInputName)
                    let output = try detModel.prediction(from: input)
                    if let mla = output.featureValue(for: outName)?.multiArrayValue {
                        let decoded = Self.decodeItems(mla,
                                                       includePersons: !settings.poseEnabled,
                                                       minConfidence: minConfidence,
                                                       letterbox: letterbox)
                        detCount = decoded.count
                        rawTargets += decoded
                    } else {
                        inferenceError = "det输出类型异常"
                    }
                } catch {
                    inferenceError = "det:\(error.localizedDescription)"
                }
            }
            self.report(detCount, poseCount, inferenceError)

            let costMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // 3) 尺寸估算：以最高的人物框为参照（同深度平面假设）
            let personBox = rawTargets
                .filter { $0.isPerson }
                .max { $0.box.height < $1.box.height }?.box
            let withSize = rawTargets.map { target -> RawTarget in
                guard !target.isPerson else { return target }
                let est = SizeEstimator.estimate(objectBox: target.box, personBox: personBox)
                return RawTarget(box: target.box, conf: target.conf, label: target.label,
                                 isPerson: target.isPerson, state: target.state,
                                 faceVisible: target.faceVisible,
                                 estimatedHeightCm: est.estimatedHeightCm)
            }

            let persons = self.tracker.assign(withSize)
            DispatchQueue.main.async { completion(persons, costMs) }
        }
    }

    // MARK: - 模型加载

    private static func loadModel(_ name: String) -> MLModel? {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return try? MLModel(contentsOf: url, configuration: config)
        }
        // 设备端必须先编译：mlmodel/mlpackage 直载会失败
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodel") {
            guard let compiled = try? MLModel.compileModel(at: url) else { return nil }
            return try? MLModel(contentsOf: compiled, configuration: config)
        }
        if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            guard let compiled = try? MLModel.compileModel(at: url) else { return nil }
            return try? MLModel(contentsOf: compiled, configuration: config)
        }
        return nil
    }

    private static func featureProvider(_ pixelBuffer: CVPixelBuffer, inputName: String) throws -> MLFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)])
    }

    private static let ciContext = CIContext(options: [.cacheIntermediates: false])

    /// 预处理：摄像头帧 → 640×640 letterbox（等比缩放 + 居中黑边）
    /// 模型仅接受固定 640×640 输入，直接送原始 1280×720 会被 CoreML 拒绝
    private static func makeLetterboxedBuffer(_ source: CVPixelBuffer) -> (CVPixelBuffer, Letterbox)? {
        let srcW = CVPixelBufferGetWidth(source)
        let srcH = CVPixelBufferGetHeight(source)
        guard srcW > 0, srcH > 0 else { return nil }
        let target = Int(inputSize)

        let scale = min(Double(target) / Double(srcW), Double(target) / Double(srcH))
        let contentW = Double(srcW) * scale
        let contentH = Double(srcH) * scale
        let padX = (Double(target) - contentW) / 2
        let padY = (Double(target) - contentH) / 2

        var out: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferWidthKey: target,
            kCVPixelBufferHeightKey: target,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, target, target,
                                  kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &out) == kCVReturnSuccess,
              let output = out else { return nil }

        // 先清黑（未填充区域必须是黑边，而非未初始化内存）
        CVPixelBufferLockBaseAddress(output, [])
        if let base = CVPixelBufferGetBaseAddress(output) {
            memset(base, 0, CVPixelBufferGetDataSize(output))
        }
        CVPixelBufferUnlockBaseAddress(output, [])

        let image = CIImage(cvPixelBuffer: source)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: padX, y: padY))
        ciContext.render(image, to: output,
                         bounds: CGRect(x: 0, y: 0, width: target, height: target),
                         colorSpace: CGColorSpaceCreateDeviceRGB())
        return (output, Letterbox(contentW: contentW, contentH: contentH, padX: padX, padY: padY))
    }

    // MARK: - 解码（内部可见，供单元测试）

    /// 检测模型 [1, 84, 8400]：4 框 + 80 类分数；白名单过滤（人物按需）
    static func decodeItems(_ mla: MLMultiArray, includePersons: Bool,
                            minConfidence: Float, letterbox: Letterbox = .identity) -> [RawTarget] {
        var result: [RawTarget] = []
        for candidate in candidates(mla, labelCount: 80, letterbox: letterbox) {
            let isPerson = candidate.classIndex == 0
            if isPerson && !includePersons { continue }
            let label = cocoLabels[candidate.classIndex]
            if !isPerson && !householdItems.contains(label) { continue }
            let floor: Float = isPerson ? max(minConfidence, personFloor) : max(minConfidence, itemFloor)
            guard candidate.score >= floor else { continue }
            result.append(RawTarget(box: candidate.box, conf: candidate.score, label: label,
                                    isPerson: isPerson, state: .unknown,
                                    faceVisible: false, estimatedHeightCm: nil))
        }
        return result
    }

    /// 姿态模型 [1, 56, 8400]：通道 0-3 框、4 置信度、5-55 为 17 关键点(x,y,vis)
    /// 注意：pose 张量不能用通用 argmax（关键点坐标会被误当类别分数），专用通道读取
    static func decodePose(_ mla: MLMultiArray, minConfidence: Float,
                           letterbox: Letterbox = .identity) -> [RawTarget] {
        let shape = mla.shape.map { $0.intValue }
        guard shape.count == 3, shape[1] == 56 else { return [] }   // 17 关键点模型

        let count = shape[2]
        let channelStride = mla.strides[1].intValue
        let indexStride = mla.strides[2].intValue

        var result: [RawTarget] = []
        for i in 0..<count {
            let base = i * indexStride
            let score = mla[base + 4 * channelStride].floatValue
            guard score >= max(minConfidence, personFloor) else { continue }

            let cx = mla[base].doubleValue
            let cy = mla[base + channelStride].doubleValue
            let w = mla[base + 2 * channelStride].doubleValue
            let h = mla[base + 3 * channelStride].doubleValue
            guard let box = boxFromModelPixels(cx: cx, cy: cy, w: w, h: h, letterbox: letterbox) else { continue }

            let keypoints = decodeKeypoints(mla, channelStride: channelStride, index: base, letterbox: letterbox)
            let pose = PoseClassifier.classify(keypoints: keypoints)
            result.append(RawTarget(box: box, conf: score, label: "person",
                                    isPerson: true, state: pose.state,
                                    faceVisible: pose.faceVisible, estimatedHeightCm: nil))
        }
        return result
    }

    private static func decodeKeypoints(_ mla: MLMultiArray,
                                        channelStride: Int,
                                        index: Int,
                                        letterbox: Letterbox) -> [Keypoint] {
        var keypoints: [Keypoint] = []
        keypoints.reserveCapacity(17)
        for k in 0..<17 {
            let px = mla[index + (5 + 3 * k) * channelStride].doubleValue
            let py = mla[index + (6 + 3 * k) * channelStride].doubleValue
            let vis = mla[index + (7 + 3 * k) * channelStride].floatValue
            // 还原 letterbox 变换（等比缩放，比例关系保持，姿态几何判定不受影响）
            let x = (px - letterbox.padX) / letterbox.contentW
            let y = (py - letterbox.padY) / letterbox.contentH
            keypoints.append(Keypoint(x: x, y: y, visible: vis >= 0.5))
        }
        return keypoints
    }

    // MARK: - 坐标还原

    /// 模型像素坐标（640 空间，原点左上）→ 归一化框（原点左下，与 CoordinateMapper 一致）
    /// YOLO 输出 cy 自顶向下，而下游坐标映射按 Vision 左下原点处理，此处必须翻转 Y
    private static func boxFromModelPixels(cx: Double, cy: Double, w: Double, h: Double,
                                           letterbox: Letterbox) -> CGRect? {
        let xNorm = (cx - w / 2 - letterbox.padX) / letterbox.contentW
        let yTopNorm = (cy - h / 2 - letterbox.padY) / letterbox.contentH
        let wNorm = w / letterbox.contentW
        let hNorm = h / letterbox.contentH
        guard wNorm > 0, hNorm > 0 else { return nil }
        return clamp(CGRect(x: xNorm, y: 1 - yTopNorm - hNorm, width: wNorm, height: hNorm))
    }

    // MARK: - 张量遍历

    struct Candidate {
        let box: CGRect
        let score: Float
        let classIndex: Int
    }

    /// 遍历 [1, C, N] 张量的 N 个候选，返回最高分类别（检测模型专用）
    private static func candidates(_ mla: MLMultiArray, labelCount: Int,
                                   letterbox: Letterbox) -> [Candidate] {
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
        let classCount = channels - 4
        guard classCount > 0 else { return [] }

        var result: [Candidate] = []
        for i in 0..<count {
            let base = i * indexStride
            let cx = mla[base].doubleValue
            let cy = mla[base + channelStride].doubleValue
            let w = mla[base + 2 * channelStride].doubleValue
            let h = mla[base + 3 * channelStride].doubleValue
            guard w > 2, h > 2 else { continue }   // 退化框（模型输入像素坐标）

            var bestClass = -1
            var bestScore = Float(0)
            for c in 0..<classCount {
                let s = mla[base + (4 + c) * channelStride].floatValue
                if s > bestScore { bestScore = s; bestClass = c }
            }
            guard bestClass >= 0 else { continue }
            guard let box = boxFromModelPixels(cx: cx, cy: cy, w: w, h: h, letterbox: letterbox) else { continue }
            result.append(Candidate(box: box, score: bestScore, classIndex: bestClass))
        }
        return result
    }

    /// 越界/过小框：收敛到画面内并剔除无效框
    private static func clamp(_ box: CGRect) -> CGRect? {
        let clamped = box.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard !clamped.isNull, clamped.width > 0.03, clamped.height > 0.03 else { return nil }
        return clamped
    }
}

// MARK: - 跟踪器（输入升级为 RawTarget）

/// 轻量跟踪：IoU 优先 + 中心距离兜底的双级匹配，
/// 检测框 EMA 平滑 + 线性速度外推 + 短暂遮挡滑行，保证目标 ID 跨帧稳定
final class PersonTracker {

    private struct Track {
        let id: UUID
        var rect: CGRect
        var velocity: CGSize
        var confidence: Float
        var label: String
        var isPerson: Bool
        var state: PersonState
        var faceVisible: Bool
        var estimatedHeightCm: Double?
        var misses: Int
    }

    private var tracks: [Track] = []
    private let smoothAlpha: CGFloat = 0.6
    private let matchThreshold: CGFloat = 0.14
    private let iouMatchThreshold: CGFloat = 0.25
    /// 遮挡滑行帧数：仅 1 帧，避免"幽灵框"
    private let maxCoastMisses = 1

    func assign(_ targets: [RawTarget]) -> [DetectedPerson] {
        for index in tracks.indices {
            tracks[index].rect = tracks[index].rect.offsetBy(dx: tracks[index].velocity.width,
                                                             dy: tracks[index].velocity.height)
        }

        var targetUsed = [Bool](repeating: false, count: targets.count)
        var trackMatched = [Bool](repeating: false, count: tracks.count)
        var result: [DetectedPerson] = []
        result.reserveCapacity(targets.count + tracks.count)

        for (tIdx, target) in targets.enumerated() {
            var bestIndex = -1
            var bestIoU = iouMatchThreshold
            for (kIdx, track) in tracks.enumerated() where !trackMatched[kIdx] {
                let score = Self.iou(track.rect, target.box)
                if score > bestIoU { bestIoU = score; bestIndex = kIdx }
            }
            if bestIndex >= 0 {
                targetUsed[tIdx] = true
                trackMatched[bestIndex] = true
                result.append(merge(index: bestIndex, target))
            }
        }

        for (tIdx, target) in targets.enumerated() where !targetUsed[tIdx] {
            let center = CGPoint(x: target.box.midX, y: target.box.midY)
            var bestIndex = -1
            var bestDistance = matchThreshold
            for (kIdx, track) in tracks.enumerated() where !trackMatched[kIdx] {
                let distance = hypot(track.rect.midX - center.x, track.rect.midY - center.y)
                if distance < bestDistance { bestDistance = distance; bestIndex = kIdx }
            }
            if bestIndex >= 0 {
                trackMatched[bestIndex] = true
                result.append(merge(index: bestIndex, target))
            } else {
                result.append(Self.make(target))
            }
        }

        for (kIdx, track) in tracks.enumerated() where !trackMatched[kIdx] {
            var coasted = track
            coasted.misses += 1
            if coasted.misses <= maxCoastMisses {
                coasted.confidence *= 0.8
                tracks[kIdx] = coasted
                result.append(Self.output(coasted))
            }
        }

        tracks = result.map { person in
            if let index = tracks.firstIndex(where: { $0.id == person.id }) {
                return tracks[index]
            }
            return Track(id: person.id, rect: person.box, velocity: .zero,
                         confidence: person.confidence, label: person.label,
                         isPerson: person.isPerson, state: person.state ?? .unknown,
                         faceVisible: person.faceVisible ?? false,
                         estimatedHeightCm: person.estimatedHeightCm, misses: 0)
        }
        return result
    }

    private func merge(index: Int, _ target: RawTarget) -> DetectedPerson {
        var track = tracks[index]
        let previousCenter = CGPoint(x: track.rect.midX, y: track.rect.midY)
        let box = CGRect(
            x: track.rect.minX + (target.box.minX - track.rect.minX) * smoothAlpha,
            y: track.rect.minY + (target.box.minY - track.rect.minY) * smoothAlpha,
            width: track.rect.width + (target.box.width - track.rect.width) * smoothAlpha,
            height: track.rect.height + (target.box.height - track.rect.height) * smoothAlpha
        )
        let newCenter = CGPoint(x: box.midX, y: box.midY)
        let residual = CGSize(width: newCenter.x - previousCenter.x,
                              height: newCenter.y - previousCenter.y)
        track.velocity = CGSize(width: track.velocity.width * 0.5 + residual.width * 0.5,
                                height: track.velocity.height * 0.5 + residual.height * 0.5)
        track.rect = box
        track.confidence = target.conf
        track.label = target.label
        track.isPerson = target.isPerson
        track.state = target.state
        track.faceVisible = target.faceVisible
        track.estimatedHeightCm = target.estimatedHeightCm
        track.misses = 0
        tracks[index] = track
        return Self.output(track)
    }

    private static func make(_ target: RawTarget) -> DetectedPerson {
        // 非人物不带姿态/人脸字段（与 output() 保持一致）
        DetectedPerson(id: UUID(), box: target.box, confidence: target.conf,
                       label: target.label, isPerson: target.isPerson,
                       state: target.isPerson ? target.state : nil,
                       faceVisible: target.isPerson ? target.faceVisible : nil,
                       estimatedHeightCm: target.isPerson ? nil : target.estimatedHeightCm)
    }

    private static func output(_ track: Track) -> DetectedPerson {
        DetectedPerson(id: track.id, box: track.rect, confidence: track.confidence,
                       label: track.label, isPerson: track.isPerson,
                       state: track.isPerson ? track.state : nil,
                       faceVisible: track.isPerson ? track.faceVisible : nil,
                       estimatedHeightCm: track.isPerson ? nil : track.estimatedHeightCm)
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
