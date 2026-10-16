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

/// 人物检测器
/// - 默认：Apple Vision 内置人体检测（零依赖，所有设备可用）
/// - 可选：Bundle 内捆绑开源模型（如 Ultralytics YOLO11n 导出的 CoreML，资源名 yolo11n）
///   时自动切换 VNCoreMLRequest（ANE 加速），内置 person 类别过滤 + IoU NMS
/// 两条路径输出统一归一化框，下游（坐标映射/瞄准/绘制）完全无感
final class PersonDetector {

    /// 可选开源模型资源名（存在 .mlmodelc / .mlmodel 时启用）
    static let coreMLModelName = "yolo11n"

    private let request: VNRequest
    private let usesCoreML: Bool
    private let queue = DispatchQueue(label: "funaim.inference", qos: .userInteractive)
    private var tracker = PersonTracker()

    init() {
        if let modelURL = Self.locateCoreMLModel(),
           let vnModel = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) {
            let coreMLRequest = VNCoreMLRequest(model: vnModel)
            coreMLRequest.cropAndScaleOption = .scaleFit
            request = coreMLRequest
            usesCoreML = true
        } else {
            let visionRequest = VNDetectHumanRectanglesRequest()
            visionRequest.upperBodyOnly = false   // 全身 + 半身均检出
            request = visionRequest
            usesCoreML = false
        }
    }

    /// 异步检测，completion 回调主线程
    func detect(pixelBuffer: CVPixelBuffer,
                minConfidence: Float,
                completion: @escaping ([DetectedPerson], Double) -> Void) {
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

            var raw = Self.parse(request: request, usesCoreML: self?.usesCoreML ?? false)
                .filter { $0.conf >= minConfidence }
                .compactMap(Self.clamp)
            if self?.usesCoreML == true {
                raw = Self.nms(raw, iouThreshold: 0.45)
            }

            let persons = self?.tracker.assign(raw) ?? []
            DispatchQueue.main.async { completion(persons, costMs) }
        }
    }

    // MARK: - 模型加载

    private static func locateCoreMLModel() -> URL? {
        if let url = Bundle.main.url(forResource: coreMLModelName, withExtension: "mlmodelc") {
            return url
        }
        if let url = Bundle.main.url(forResource: coreMLModelName, withExtension: "mlmodel") {
            return try? MLModel.compileModel(at: url)   // 一次性编译到临时目录
        }
        return nil
    }

    // MARK: - 结果解析

    private static func parse(request: VNRequest, usesCoreML: Bool) -> [(box: CGRect, conf: Float)] {
        if usesCoreML {
            // YOLO 类模型：VNRecognizedObjectObservation，按类别名过滤
            let observations = (request.results as? [VNRecognizedObjectObservation]) ?? []
            return observations
                .filter { observation in
                    observation.labels.contains { $0.identifier.lowercased().contains("person") }
                }
                .map { (box: $0.boundingBox, conf: $0.confidence) }
        }
        // Vision 内置路径：VNHumanObservation
        let observations = (request.results as? [VNHumanObservation]) ?? []
        return observations.map { (box: $0.boundingBox, conf: $0.confidence) }
    }

    /// Vision / 模型偶发输出越界框：收敛到画面内并剔除无效框
    private static func clamp(_ detection: (box: CGRect, conf: Float)) -> (box: CGRect, conf: Float)? {
        let clamped = detection.box.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard !clamped.isNull, clamped.width > 0.01, clamped.height > 0.01 else { return nil }
        return (box: clamped, conf: detection.conf)
    }

    /// IoU 非极大值抑制（模型未内置 NMS 时兜底）
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

/// 轻量跟踪：最近邻匹配分配跨帧稳定 ID + 检测框 EMA 平滑（抑制逐帧抖动）
private final class PersonTracker {

    private struct Track {
        let id: UUID
        let rect: CGRect
    }

    private var tracks: [Track] = []
    /// 平滑系数：越大越贴测量值（0.6 ≈ 抑制抖动且延迟极小）
    private let smoothAlpha: CGFloat = 0.6
    /// 归一化坐标下的匹配阈值
    private let matchThreshold: CGFloat = 0.14

    func assign(_ detections: [(box: CGRect, conf: Float)]) -> [DetectedPerson] {
        var used = [Bool](repeating: false, count: tracks.count)
        var result: [DetectedPerson] = []
        result.reserveCapacity(detections.count)

        for detection in detections {
            let center = CGPoint(x: detection.box.midX, y: detection.box.midY)
            var matchIndex = -1
            var bestDistance = matchThreshold
            for (index, track) in tracks.enumerated() where !used[index] {
                let trackCenter = CGPoint(x: track.rect.midX, y: track.rect.midY)
                let distance = hypot(trackCenter.x - center.x, trackCenter.y - center.y)
                if distance < bestDistance {
                    bestDistance = distance
                    matchIndex = index
                }
            }

            let id: UUID
            let box: CGRect
            if matchIndex >= 0 {
                used[matchIndex] = true
                id = tracks[matchIndex].id
                // 指数平滑：位置与尺寸同时收敛，消除框的逐帧跳变
                let previous = tracks[matchIndex].rect
                box = CGRect(
                    x: previous.minX + (detection.box.minX - previous.minX) * smoothAlpha,
                    y: previous.minY + (detection.box.minY - previous.minY) * smoothAlpha,
                    width: previous.width + (detection.box.width - previous.width) * smoothAlpha,
                    height: previous.height + (detection.box.height - previous.height) * smoothAlpha
                )
            } else {
                id = UUID()
                box = detection.box
            }
            result.append(DetectedPerson(id: id, box: box, confidence: detection.conf))
        }

        tracks = result.map { Track(id: $0.id, rect: $0.box) }
        return result
    }
}
