import CoreGraphics
import Foundation

/// Vision 归一化坐标（原点左下）→ 预览视图坐标（原点左上）
/// 预览层使用 resizeAspectFill（等比填充 + 居中裁剪），此处做同构逆映射
struct CoordinateMapper {

    let viewSize: CGSize
    let bufferSize: CGSize   // 实际送入推理的像素缓冲尺寸（竖屏）

    func map(_ normalizedRect: CGRect) -> CGRect {
        guard viewSize.width > 0, viewSize.height > 0,
              bufferSize.width > 0, bufferSize.height > 0 else { return .zero }

        let scale = max(viewSize.width / bufferSize.width, viewSize.height / bufferSize.height)
        let scaledW = bufferSize.width * scale
        let scaledH = bufferSize.height * scale
        let offsetX = (scaledW - viewSize.width) / 2
        let offsetY = (scaledH - viewSize.height) / 2

        // Y 轴翻转：Vision 坐标原点在左下，视图原点在左上
        let x1 = normalizedRect.minX * scaledW - offsetX
        let x2 = normalizedRect.maxX * scaledW - offsetX
        let yTop = (1 - normalizedRect.maxY) * scaledH - offsetY
        let yBottom = (1 - normalizedRect.minY) * scaledH - offsetY

        return CGRect(x: x1, y: yTop, width: x2 - x1, height: yBottom - yTop)
    }

    /// 单点映射（用于头部瞄准点）
    func mapPoint(_ point: CGPoint) -> CGPoint {
        guard viewSize.width > 0, viewSize.height > 0,
              bufferSize.width > 0, bufferSize.height > 0 else { return .zero }
        let scale = max(viewSize.width / bufferSize.width, viewSize.height / bufferSize.height)
        let scaledW = bufferSize.width * scale
        let scaledH = bufferSize.height * scale
        let offsetX = (scaledW - viewSize.width) / 2
        let offsetY = (scaledH - viewSize.height) / 2
        return CGPoint(x: point.x * scaledW - offsetX,
                       y: (1 - point.y) * scaledH - offsetY)
    }
}
