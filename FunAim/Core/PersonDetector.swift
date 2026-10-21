import CoreML
import CoreVideo
import Foundation

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

    init() {
        detModel = Self.loadModel(Self.detModelName)
        if let m = detModel {
            detInputName = m.modelDescription.inputDescriptionsByName.first?.key ?? "image"
            detOutputName = m.modelDescription.outputDescriptionsByName
                .first { $0.value.type == .multiArray }?.key
        }
        poseModel = Self.loadModel(Self.poseModelName)
        if let m = poseModel {
            poseInputName = m.modelDescription.inputDescriptionsByName.first?.key ?? "image"
            poseOutputName = m.modelDescription.outputDescriptionsByName
                .first { $0.value.type == .multiArray }?.key
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

            // 1) 姿态模型：人物框 + 17 关键点（人物唯一来源，避免双模型重复框）
            if settings.poseEnabled, let poseModel = self.poseModel, let outName = self.poseOutputName {
                do {
                    let input = try Self.featureProvider(pixelBuffer, inputName: self.poseInputName)
                    let output = try poseModel.prediction(from: input)
                    if let mla = output.featureValue(for: outName)?.multiArrayValue {
                        rawTargets += Self.decodePose(mla, minConfidence: minConfidence)
                    }
                } catch { }
            }

            // 2) 检测模型：生活物品白名单；姿态关闭时 person 也由此模型给出（无姿态信息）
            if settings.itemsEnabled || !settings.poseEnabled,
               let detModel = self.detModel, let outName = self.detOutputName {
                do {
                    let input = try Self.featureProvider(pixelBuffer, inputName: self.detInputName)
                    let output = try detModel.prediction(from: input)
                    if let mla = output.featureValue(for: outName)?.multiArrayValue {
                        rawTargets += Self.decodeItems(mla,
                                                       includePersons: !settings.poseEnabled,
                                                       minConfidence: minConfidence)
                    }
                } catch { }
            }

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
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodel") {
            guard let compiled = try? MLModel.compileModel(at: url) else { return nil }
            return try? MLModel(contentsOf: compiled, configuration: config)
        }
        if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            return try? MLModel(contentsOf: url, configuration: config)
        }
        return nil
    }

    private static func featureProvider(_ pixelBuffer: CVPixelBuffer, inputName: String) throws -> MLFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)])
    }

    // MARK: - 解码（内部可见，供单元测试）

    /// 检测模型 [1, 84, 8400]：4 框 + 80 类分数；白名单过滤（人物按需）
    static func decodeItems(_ mla: MLMultiArray, includePersons: Bool, minConfidence: Float) -> [RawTarget] {
        var result: [RawTarget] = []
        for candidate in candidates(mla, labelCount: 80) {
            let isPerson = candidate.classIndex == 0
            if isPerson && !includePersons { continue }
            let label = cocoLabels[candidate.classIndex]
            if !isPerson && !householdItems.contains(label) { continue }
            let floor: Float = isPerson ? max(minConfidence, personFloor) : max(minConfidence, itemFloor)
            guard candidate.score >= floor else { continue }
            result.append(RawTarget(box: candidate.box, conf: candidate.score, label: label,
                                    isPerson: isPerson, state: .unknown,
                                    faceVisible: nil, estimatedHeightCm: nil))
        }
        return result
    }

    /// 姿态模型 [1, 56, 8400]：通道 0-3 框、4 置信度、5-55 为 17 关键点(x,y,vis)
    /// 注意：pose 张量不能用通用 argmax（关键点坐标会被误当类别分数），专用通道读取
    static func decodePose(_ mla: MLMultiArray, minConfidence: Float) -> [RawTarget] {
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
            guard let box = clamp(CGRect(x: (cx - w / 2) / Double(inputSize),
                                         y: (cy - h / 2) / Double(inputSize),
                                         width: w / Double(inputSize),
                                         height: h / Double(inputSize))) else { continue }

            let keypoints = decodeKeypoints(mla, channelStride: channelStride, index: base)
            let pose = PoseClassifier.classify(keypoints: keypoints)
            result.append(RawTarget(box: box, conf: score, label: "person",
                                    isPerson: true, state: pose.state,
                                    faceVisible: pose.faceVisible, estimatedHeightCm: nil))
        }
        return result
    }

    private static func decodeKeypoints(_ mla: MLMultiArray,
                                        channelStride: Int,
                                        index: Int) -> [Keypoint] {
        var keypoints: [Keypoint] = []
        keypoints.reserveCapacity(17)
        for k in 0..<17 {
            let x = mla[index + (5 + 3 * k) * channelStride].doubleValue / Double(inputSize)
            let y = mla[index + (6 + 3 * k) * channelStride].doubleValue / Double(inputSize)
            let vis = mla[index + (7 + 3 * k) * channelStride].floatValue
            keypoints.append(Keypoint(x: x, y: y, visible: vis >= 0.5))
        }
        return keypoints
    }

    // MARK: - 张量遍历

    struct Candidate {
        let box: CGRect
        let score: Float
        let classIndex: Int
    }

    /// 遍历 [1, C, N] 张量的 N 个候选，返回最高分类别（检测模型专用）
    private static func candidates(_ mla: MLMultiArray, labelCount: Int) -> [Candidate] {
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
            guard let box = clamp(CGRect(x: (cx - w / 2) / Double(inputSize),
                                         y: (cy - h / 2) / Double(inputSize),
                                         width: w / Double(inputSize),
                                         height: h / Double(inputSize))) else { continue }
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
        DetectedPerson(id: UUID(), box: target.box, confidence: target.conf,
                       label: target.label, isPerson: target.isPerson,
                       state: target.state, faceVisible: target.faceVisible,
                       estimatedHeightCm: target.estimatedHeightCm)
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
