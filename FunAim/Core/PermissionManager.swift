import AVFoundation
import UIKit

/// 摄像头权限管理：启动第一时间申请，失败时提供跳转系统设置的引导
final class PermissionManager: ObservableObject {

    enum CameraStatus {
        case undetermined   // 尚未询问
        case denied         // 被拒绝 / 受限
        case authorized     // 已授权
    }

    @Published private(set) var status: CameraStatus = PermissionManager.current()

    static func current() -> CameraStatus {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return .authorized
        case .notDetermined:
            return .undetermined
        default:
            return .denied
        }
    }

    /// 弹出系统权限申请框（仅未询问时有效）
    func request() {
        guard status == .undetermined else { return }
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                self?.status = granted ? .authorized : .denied
            }
        }
    }

    /// 从系统设置返回后重新读取权限状态
    func refresh() {
        status = PermissionManager.current()
    }

    /// 跳转本应用的系统设置页，引导用户手动开启摄像头
    func openSystemSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }
}
