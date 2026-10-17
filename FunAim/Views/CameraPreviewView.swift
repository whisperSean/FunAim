import SwiftUI
import AVFoundation

/// 相机预览层封装：旋转角与数据管线由同一来源驱动（ContentView 传入），
/// 避免横屏锁定启动时预览层停留在竖屏方向导致的"画面不像横屏"问题。
struct CameraPreviewView: UIViewRepresentable {

    let session: AVCaptureSession
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
