import CoreML
import CoreVideo
import Foundation
import Vision

/// 单个人物检测结果
/// box 为归一化坐标（原点左下，与 Vision 一致）
struct DetectedPerson {
    let id: UUID
    let box: CGRect
    let confidence: Float
}

/// 人物检测器：统一使用 CI 打包的 YOLO11n CoreML 模型
/// 单一可靠路径：仅识别人物（person 类），VNCoreMLRequest 自动调度 ANE/GPU，
/// 本地完成置信度过滤 + 越界收敛 + IoU NMS，输出归一化框供下游管线使用
final class PersonDetector {

    static let coreMLModelName = "yolo11n"

    private let request: VNCoreMLRequest?
    private let queue = DispatchQueue(label: "funaim.inference", qos: .userInteractive)
    private var tracker = PersonTracker()

    init() {
        guard let modelURL = Self.locateCoreMLModel(),
              let vnModel = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else {
            request = nil
            return
        }
        request = VNCoreMLRequest(model: vnModel)
    }

    /// 推理预热：消除首次 perform 的冷启动开销（会话启动时调用一次）
    func warmup() {
        queue.async { [request] in
            guard let request else { return }
            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
            guard let pb = pixelBuffer else { return }
            let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
            try? handler.perform([request])
        }
    }

    /// 异步检测，completion 回调主线程
    func detect(pixelBuffer: CVPixelBuffer,
                minConfidence: Float,
                completion: @escaping ([DetectedPerson], Double) -> Void) {
        guard let request else {
            DispatchQueue.main.async { completion([], 0) }
            return
        }
        queue.async { [request, weak self] in
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
            let t0 = CFAbsoluteTimeGetCurrent()
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async { completion([], 0) }
                return
            }
            let costMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // 仅保留 person 类，再做置信度过滤与越界收敛
            let observations = (request.results as? [VNRecognizedObjectObservation]) ?? []
            let raw = observations
                .filter { $0.labels.contains { $0.identifier.lowercased().contains("person") } }
                .filter { $0.confidence >= minConfidence }
                .map { (box: $0.boundingBox, conf: $0.confidence) }
                .compactMap(Self.clamp)
            let filtered = Self.nms(raw, iouThreshold: 0.45)

            let persons = self?.tracker.assign(filtered) ?? []
            DispatchQueue.main.async { completion(persons, costMs) }
        }
    }

    // MARK: - 模型加载

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

    /// 模型偶发输出越界框：收敛到画面内并剔除无效框
    private static func clamp(_ detection: (box: CGRect, conf: Float)) -> (box: CGRect, conf: Float)? {
        let clamped = detection.box.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard !clamped.isNull, clamped.width > 0.01, clamped.height > 0.01 else { return nil }
        return (box: clamped, conf: detection.conf)
    }

    /// IoU 非极大值抑制
    private static func nms(_ input: [(box: CGRect, conf: Float)], iouThreshold: Float) -> [(box: CGRect, conf: Float)] {
        let sorted = input.sorted { $0.conf > $1.conf }
        var keep: [(box: CGRect, conf: Float)] = []
        for candidate in sorted where keep.allSatisfy({ iou($0.box, candidate.box) < iouThreshold }) {
            keep.append(candidate)
        }
        return keep
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
        var misses: Int
    }

    private var tracks: [Track] = []
    /// 平滑系数：越大越贴测量值（0.6 ≈ 抑制抖动且延迟极小）
    private let smoothAlpha: CGFloat = 0.6
    /// 中心距离兜底匹配阈值（归一化坐标）
    private let matchThreshold: CGFloat = 0.14
    /// IoU 匹配阈值
    private let iouMatchThreshold: CGFloat = 0.25
    /// 遮挡滑行帧数（推理帧）：期间按速度外推输出，保持锁定稳定
    private let maxCoastMisses = 2

    func assign(_ detections: [(box: CGRect, conf: Float)]) -> [DetectedPerson] {
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
                result.append(DetectedPerson(id: UUID(), box: detection.box, confidence: detection.conf))
            }
        }

        // 4) 未匹配轨迹：按速度滑行输出（置信度衰减），超限移除
        for (tIndex, track) in tracks.enumerated() where !trackMatched[tIndex] {
            var coasted = track
            coasted.misses += 1
            if coasted.misses <= maxCoastMisses {
                coasted.confidence *= 0.8
                tracks[tIndex] = coasted
                result.append(DetectedPerson(id: coasted.id, box: coasted.rect, confidence: coasted.confidence))
            }
        }

        // 5) 用本帧结果重建轨迹表（merge/滑行均已把最新值写回）
        tracks = result.map { person in
            if let index = tracks.firstIndex(where: { $0.id == person.id }) {
                return tracks[index]
            }
            return Track(id: person.id, rect: person.box, velocity: .zero,
                         confidence: person.confidence, misses: 0)
        }
        return result
    }

    /// EMA 更新 + 速度估计
    private func merge(index: Int, _ detection: (box: CGRect, conf: Float)) -> DetectedPerson {
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
        track.misses = 0
        tracks[index] = track
        return DetectedPerson(id: track.id, box: box, confidence: detection.conf)
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
