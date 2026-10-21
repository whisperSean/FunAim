import SwiftUI
import AVFoundation

/// 相机预览层封装
/// 方向策略（最终版）：预览层与数据输出管线由同一旋转角驱动（来源：UI 界面方向）。
///
/// 关键修复：AVFoundation 会在【视图挂载】和【session.startRunning()】两个时机用
/// 系统默认角度重置连接（横屏锁定启动时默认是竖屏 90°，导致首帧视角错误）。
/// 因此这里在 didMoveToWindow / layoutSubviews 时重新断言角度，
/// CameraService 则在 startRunning() 之后重新断言数据管线角度——
/// 系统重置发生在哪里，我们就跟到哪里重设。
struct CameraPreviewView: UIViewRepresentable {

    let session: AVCaptureSession
    /// 与数据管线一致的旋转角
    var rotationAngle: CGFloat = 0

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        view.rotationAngle = rotationAngle
        view.onRotationChange = { _ in }   // 兼容保留
        DispatchQueue.main.async { view.assertRotation() }
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.rotationAngle = rotationAngle   // didSet 内自动断言
    }

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        var onRotationChange: ((CGFloat) -> Void)?

        /// 目标旋转角：赋值即断言（预览层连接）
        var rotationAngle: CGFloat = 0 {
            didSet { assertRotation() }
        }

        override func didMoveToWindow() {
            super.didMoveToWindow()
            assertRotation()   // 挂载到窗口：系统可能刚重置过连接角度
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            assertRotation()   // 布局变化（含界面方向切换后的重排）
        }

        func assertRotation() {
            guard let connection = videoPreviewLayer.connection, window != nil else { return }
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(rotationAngle) {
                    if connection.videoRotationAngle != rotationAngle {
                        connection.videoRotationAngle = rotationAngle
                    }
                }
            } else if connection.isVideoOrientationSupported {
                let target = CameraRotation.avOrientation(for: rotationAngle)
                if connection.videoOrientation != target {
                    connection.videoOrientation = target
                }
            }
        }
    }
}
