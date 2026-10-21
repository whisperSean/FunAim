import AVFoundation
import Foundation
import UIKit

enum CameraError: LocalizedError {
    case noCamera
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .noCamera: return "未找到可用的摄像头设备"
        case .cannotAddInput: return "无法添加摄像头输入源"
        case .cannotAddOutput: return "无法添加视频输出流"
        }
    }
}

/// 相机服务：采集 + 帧分发 + FPS 统计
/// 预览层独立渲染（最高 60fps），AI 推理按"运行模式"的帧间隔执行，两者互不阻塞
final class CameraService: NSObject, ObservableObject {

    static let shared = CameraService()

    let session = AVCaptureSession()

    @Published private(set) var isRunning = false
    @Published private(set) var fps: Double = 0
    @Published private(set) var inferenceMs: Double = 0
    @Published private(set) var detectionCount = 0
    /// 实际送入推理的像素缓冲尺寸（横屏方向）
    @Published private(set) var bufferSize: CGSize = .zero
    /// 当前数据管线旋转角（供预览层同步使用）
    @Published private(set) var dataRotationAngle: CGFloat = 0
    /// 闪光灯状态
    @Published private(set) var torchOn = false
    /// 当前变焦倍数
    @Published private(set) var currentZoom: CGFloat = 1
    @Published var lastError: String?

    /// 检测结果回调（主线程）
    var onDetection: (([DetectedPerson]) -> Void)?

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "funaim.session", qos: .userInteractive)
    private let outputQueue = DispatchQueue(label: "funaim.capture", qos: .userInteractive)
    private let detector = PersonDetector()

    private var frameCounter = 0
    private var framesSinceMark = 0
    private var lastFPSMark = Date()
    private var isConfigured = false
    /// 最后已知旋转角（仅 sessionQueue 读写）
    private var lastRotationAngle: CGFloat = 0
    /// 过热降级：推理帧间隔下限（nominal/fair=1，serious=2，critical=3）
    private var thermalFloorStride = 1
    private var thermalObserver: NSObjectProtocol?
    private weak var currentDevice: AVCaptureDevice?

    private override init() {
        super.init()
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            self?.refreshThermalFloor()
        }
        refreshThermalFloor()
    }

    deinit {
        if let thermalObserver {
            NotificationCenter.default.removeObserver(thermalObserver)
        }
    }

    private func refreshThermalFloor() {
        switch ProcessInfo.processInfo.thermalState {
        case .critical: thermalFloorStride = 3
        case .serious:  thermalFloorStride = 2
        default:        thermalFloorStride = 1
        }
    }

    /// 幂等：配置并启动采集会话（后台线程执行，避免阻塞主线程）
    func configureAndStart() {
        sessionQueue.async { [weak self] in
            guard let self, !self.isRunning else { return }
            if !self.isConfigured {
                do {
                    try self.configure()
                    self.isConfigured = true
                } catch {
                    DispatchQueue.main.async { self.lastError = error.localizedDescription }
                    return
                }
            }
            self.session.startRunning()
            // startRunning 可能重置连接角度：重新断言，保证首帧方向正确
            self.applyRotationOnConnection(self.lastRotationAngle)
            self.detector.warmup()   // 推理预热，消除首帧冷启动
            DispatchQueue.main.async { self.isRunning = self.session.isRunning }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.stopRunning()
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    /// 按旋转角同步数据管线（sessionQueue 独占 lastRotationAngle，避免与 configure 竞争）
    func applyRotationAngle(_ angle: CGFloat) {
        DispatchQueue.main.async { self.dataRotationAngle = angle }
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.lastRotationAngle = angle
            self.applyRotationOnConnection(angle)
        }
    }

    private func applyRotationOnConnection(_ angle: CGFloat) {
        guard let connection = videoOutput.connection(with: .video) else { return }
        if #available(iOS 17.0, *) {
            if connection.isVideoRotationAngleSupported(angle) {
                connection.videoRotationAngle = angle
            }
        } else if connection.isVideoOrientationSupported {
            connection.videoOrientation = CameraRotation.avOrientation(for: angle)
        }
    }

    /// 双指变焦（1x~6x，越界钳制）
    func setZoom(_ factor: CGFloat) {
        let clamped = min(max(factor, 1), 6)
        sessionQueue.async { [weak self] in
            guard let self, let device = self.currentDevice else { return }
            do {
                try device.lockForConfiguration()
                device.videoZoomFactor = clamped
                device.unlockForConfiguration()
                DispatchQueue.main.async { self.currentZoom = clamped }
            } catch { }
        }
    }

    func toggleTorch() {
        setTorch(!torchOn)
    }

    func setTorch(_ on: Bool) {
        sessionQueue.async { [weak self] in
            guard let self, let device = self.currentDevice, device.hasTorch else { return }
            do {
                try device.lockForConfiguration()
                device.torchMode = on ? .on : .off
                device.unlockForConfiguration()
                DispatchQueue.main.async { self.torchOn = on }
            } catch { }
        }
    }

    // MARK: - 配置

    private func configure() throws {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CameraError.noCamera
        }

        // 优先选择 720p@60fps 格式；不支持时回退 720p@30
        let highFPS = (try? selectHighFrameRateFormat(for: device)) ?? false

        let input = try AVCaptureDeviceInput(device: device)
        currentDevice = device

        session.beginConfiguration()
        session.sessionPreset = highFPS ? .inputPriority : .hd1280x720
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            throw CameraError.cannotAddInput
        }
        session.addInput(input)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            throw CameraError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        session.commitConfiguration()
        // 按最后已知方向旋转，保证首帧方向即正确（默认 0 = 传感器原生横向）
        applyRotationOnConnection(lastRotationAngle)
    }

    /// 在设备支持范围内选取 ≤1080p 且能跑 60fps 的最优格式（优先 720p60）
    private func selectHighFrameRateFormat(for device: AVCaptureDevice) throws -> Bool {
        var bestFormat: AVCaptureDevice.Format?
        var bestScore = -1
        for format in device.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard dims.width <= 1920, dims.height <= 1080 else { continue }
            guard let maxRate = format.videoSupportedFrameRateRanges.map({ $0.maxFrameRate }).max(),
                  maxRate >= 60 else { continue }
            let score = (dims.height == 720 ? 100_000 : 0) + Int(dims.height) + Int(maxRate)
            if score > bestScore {
                bestScore = score
                bestFormat = format
            }
        }
        guard let format = bestFormat else { return false }
        try device.lockForConfiguration()
        device.activeFormat = format
        let frameDuration = CMTime(value: 1, timescale: 60)
        device.activeVideoMinFrameDuration = frameDuration
        device.activeVideoMaxFrameDuration = frameDuration
        device.unlockForConfiguration()
        return true
    }
}

// MARK: - 旋转角映射

/// 旋转工具（iOS 17 以下回退路径使用）
enum CameraRotation {
    /// 数据管线旋转角 → AVCaptureVideoOrientation
    static func avOrientation(for angle: CGFloat) -> AVCaptureVideoOrientation {
        switch angle {
        case 90:  return .portrait
        case 270: return .portraitUpsideDown
        case 180: return .landscapeLeft
        default:  return .landscapeRight
        }
    }
}

// MARK: - 采集回调

extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // 记录实际缓冲尺寸（竖屏旋转后），供坐标映射使用
        if bufferSize == .zero, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let size = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
            DispatchQueue.main.async { [weak self] in
                if self?.bufferSize == .zero { self?.bufferSize = size }
            }
        }

        // FPS 统计（约 1Hz 刷新；@Published 必须回主线程更新）
        framesSinceMark += 1
        let now = Date()
        let interval = now.timeIntervalSince(lastFPSMark)
        if interval >= 1 {
            let measured = Double(framesSinceMark) / interval
            framesSinceMark = 0
            lastFPSMark = now
            DispatchQueue.main.async { [weak self] in self?.fps = measured }
        }

        // 按"运行模式"的帧间隔送入本地推理（每帧直读设置中心，杜绝切换不生效）；
        // 过热时自动提升推理帧间隔下限
        let settings = AppSettings.shared
        frameCounter += 1
        let stride = max(settings.detectionMode.frameStride, thermalFloorStride)
        guard frameCounter % max(stride, 1) == 0,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        detector.detect(pixelBuffer: pixelBuffer, minConfidence: Float(settings.minConfidence)) { [weak self] persons, cost in
            guard let self else { return }
            self.inferenceMs = cost
            self.detectionCount = persons.count
            self.onDetection?(persons)
        }
    }
}
