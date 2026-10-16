import CoreGraphics
import Foundation

/// 已映射到视图坐标系的检测结果
struct MappedPerson {
    let person: DetectedPerson
    let viewRect: CGRect
    var center: CGPoint { CGPoint(x: viewRect.midX, y: viewRect.midY) }
    var id: UUID { person.id }
}

/// 自动瞄准引擎：目标选择 + 准星平滑追踪，纯本地几何计算
final class AimEngine: ObservableObject {

    /// 准星位置（归一化 0~1），初始位于画面中心
    @Published private(set) var reticle: CGPoint = CGPoint(x: 0.5, y: 0.5)
    @Published private(set) var targetID: UUID?
    @Published private(set) var locked = false

    /// 选择目标：距离准星最近、置信度最高者胜出；
    /// lockStrength 提供粘滞（锁住当前目标不易被抢），多目标场景防抖
    func selectTarget(from persons: [MappedPerson], viewSize: CGSize, lockStrength: Double) -> MappedPerson? {
        guard !persons.isEmpty, viewSize.width > 0, viewSize.height > 0 else {
            targetID = nil
            locked = false
            return nil
        }
        let current = reticlePoint(in: viewSize)
        let stick = CGFloat(lockStrength) * viewSize.width * 0.3

        var best: MappedPerson?
        var bestScore = CGFloat.greatestFiniteMagnitude
        for person in persons {
            let distance = hypot(person.center.x - current.x, person.center.y - current.y)
            let confidencePenalty = (1 - CGFloat(person.person.confidence)) * viewSize.width * 0.25
            let stickyBonus = (person.id == targetID ? -stick : 0)
            let score = distance + confidencePenalty + stickyBonus
            if score < bestScore {
                bestScore = score
                best = person
            }
        }
        targetID = best?.id
        locked = (best != nil)
        return best
    }

    /// 指数平滑移动准星；tracking ∈ (0.02, 1]，越大跟得越紧
    func moveReticle(toward point: CGPoint, in viewSize: CGSize, tracking: Double) {
        guard viewSize.width > 0, viewSize.height > 0 else { return }
        let target = CGPoint(x: point.x / viewSize.width, y: point.y / viewSize.height)
        let alpha = CGFloat(min(max(tracking, 0.02), 1))
        reticle = CGPoint(
            x: reticle.x + (target.x - reticle.x) * alpha,
            y: reticle.y + (target.y - reticle.y) * alpha
        )
    }

    func reticlePoint(in viewSize: CGSize) -> CGPoint {
        CGPoint(x: reticle.x * viewSize.width, y: reticle.y * viewSize.height)
    }

    func reset() {
        reticle = CGPoint(x: 0.5, y: 0.5)
        targetID = nil
        locked = false
    }
}
