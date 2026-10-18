import SwiftUI
import AVFoundation

/// 相机预览层封装
/// 方向策略（重构）：预览层连接由系统自动跟随界面方向旋转——用户看到的画面永远正确；
/// 通过回调把预览层当前旋转角上报，数据输出管线复制同一角度，实现"所见即所得"，
/// 不再依赖任何设备姿态 → 角度的硬编码映射表。
struct CameraPreviewView: UIViewRepresentable {

    let session: AVCaptureSession
    /// 预览层旋转角变化回调（主线程）
    var onRotationChange: ((CGFloat) -> Void)? = nil

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        view.onRotationChange = onRotationChange
        DispatchQueue.main.async { view.reportRotation() }
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.onRotationChange = onRotationChange
        uiView.reportRotation()
    }

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        var onRotationChange: ((CGFloat) -> Void)?

        override func layoutSubviews() {
            super.layoutSubviews()
            reportRotation()   // 界面方向变化时系统会触发重新布局
        }

        func reportRotation() {
            guard let connection = videoPreviewLayer.connection else { return }
            let angle: CGFloat
            if #available(iOS 17.0, *) {
                angle = connection.videoRotationAngle
            } else {
                angle = CameraRotation.angle(of: connection.videoOrientation)
            }
            onRotationChange?(angle)
        }
    }
}
