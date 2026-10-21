import SwiftUI
import AVFoundation

/// 相机预览层封装
/// 方向策略（最终版，与系统相机同源）：
/// 界面方向（UIInterfaceOrientation，系统对"用户怎么握"的最终裁决）→ 显式设置
/// 预览层与数据输出管线的旋转角，两端永远一致；不依赖预览层"自动旋转"（该行为
/// 在横屏锁定的 SwiftUI 应用启动时不可靠），也不使用设备姿态查表。
struct CameraPreviewView: UIViewRepresentable {

    let session: AVCaptureSession
    /// 与数据管线一致的旋转角
    var rotationAngle: CGFloat = 0

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        applyAngle(to: view)
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        applyAngle(to: uiView)
    }

    private func applyAngle(to view: PreviewUIView) {
        guard let connection = view.videoPreviewLayer.connection else { return }
        if #available(iOS 17.0, *) {
            if connection.isVideoRotationAngleSupported(rotationAngle) {
                connection.videoRotationAngle = rotationAngle
            }
        } else if connection.isVideoOrientationSupported {
            connection.videoOrientation = CameraRotation.avOrientation(for: rotationAngle)
        }
    }

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
