import SwiftUI

// MARK: - 设计令牌

/// iOS 动效令牌：全局统一弹性曲线，保证动效一致、过渡丝滑
enum Motion {
    /// 标准弹性（卡片状态、内容展开、玻璃强度变化）
    static let spring = Animation.spring(response: 0.38, dampingFraction: 0.82)
    /// 快速弹性（按压、色板选中、小元素）
    static let springSnappy = Animation.spring(response: 0.28, dampingFraction: 0.72)
    /// 灵动岛同款"果冻感"曲线（胶囊形变、宽高展开）
    static let island = Animation.spring(response: 0.45, dampingFraction: 0.78)
}

/// 圆角令牌：continuous 大圆角，与 iOS 原生组件规范一致
enum Radii {
    static let card: CGFloat = 22
    static let chip: CGFloat = 16
}

// MARK: - 液态玻璃材质

/// 液态玻璃背景（克制苹果风）：毛玻璃模糊 + 极轻明暗着色 + 发丝描边 + 柔和投影
/// intensity 0~1 驱动玻璃"厚度"：着色、描边亮度与阴影强度，可由滚动偏移或应用状态动态驱动
struct LiquidGlassBackground: ViewModifier {

    var cornerRadius: CGFloat
    var intensity: Double

    func body(content: Content) -> some View {
        content
            .background(
                ZStack {
                    // 毛玻璃基底（模糊）
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .fill(.ultraThinMaterial)

                    // 玻璃体着色：极轻微的顶亮底沉，仅保留厚度暗示
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [
                                    .white.opacity(0.09 * intensity + 0.03),
                                    .clear,
                                    .black.opacity(0.04)
                                ],
                                startPoint: .top, endPoint: .bottom
                            )
                        )

                    // 统一发丝描边
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.16 + 0.12 * intensity), lineWidth: 0.5)
                }
            )
            // 柔和单层投影
            .shadow(color: .black.opacity(0.06 + 0.06 * intensity),
                    radius: 10 + 6 * intensity, y: 3 + 2 * intensity)
    }
}

extension View {
    func liquidGlass(cornerRadius: CGFloat, intensity: Double = 0.8) -> some View {
        modifier(LiquidGlassBackground(cornerRadius: cornerRadius, intensity: intensity))
    }
}

// MARK: - 交互组件

/// iOS 标准按压反馈：弹性缩放 + 轻微降透明度
struct PressableButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.95 : 1)
            .opacity(configuration.isPressed ? 0.85 : 1)
            .animation(Motion.springSnappy, value: configuration.isPressed)
    }
}

/// 呼吸脉冲：用于状态点与图标（active=false 时静止）
struct PulseEffect: ViewModifier {

    var active: Bool = true
    var scale: CGFloat = 1.35
    @State private var pulsing = false

    func body(content: Content) -> some View {
        content
            .scaleEffect(active ? (pulsing ? scale : 1) : 1)
            .opacity(active ? (pulsing ? 0.55 : 1) : 1)
            .animation(
                active ? .easeInOut(duration: 0.9).repeatForever(autoreverses: true) : .linear(duration: 0.01),
                value: pulsing
            )
            .onAppear { pulsing = true }
    }
}

// MARK: - 玻璃信息条

struct GlassChip<Content: View>: View {

    var cornerRadius: CGFloat = Radii.chip
    let content: () -> Content

    init(cornerRadius: CGFloat = Radii.chip, @ViewBuilder content: @escaping () -> Content) {
        self.cornerRadius = cornerRadius
        self.content = content
    }

    var body: some View {
        content()
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .liquidGlass(cornerRadius: cornerRadius, intensity: 0.7)
    }
}

// MARK: - 灵动岛式状态胶囊

/// 顶部状态胶囊：外观与灵动岛一致（纯黑胶囊 + 顶部镜面高光 + 细描边），
/// 随应用状态（瞄准关闭/搜索/发现/锁定）以灵动岛同款弹簧曲线形变，
/// 实现界面与灵动岛区域的自然衔接与自适应。
/// 位置位于安全区顶部（灵动岛下方），预览画面延伸至其背后，互不遮挡。
struct DynamicStatusCapsule: View {

    enum Phase: Equatable {
        case idle        // 瞄准关闭：紧凑待机
        case searching   // 瞄准开启、无目标
        case tracking    // 发现目标未锁定
        case locked      // 已锁定
    }

    let phase: Phase
    let targetCount: Int
    let fps: Int
    /// 设置面板展开态：胶囊变宽显示"功能设置"
    var isExpanded: Bool = false
    /// 点击回调（nil 时不可交互）
    var onTap: (() -> Void)? = nil

    var body: some View {
        ZStack {
            surface
            content
        }
        .frame(width: size.width, height: size.height)
        .animation(Motion.island, value: phase)
        .animation(Motion.island, value: isExpanded)
        .onTapGesture { onTap?() }
        .allowsHitTesting(onTap != nil)
    }

    // MARK: 尺寸（宽度与高度随状态形变）

    private var size: (width: CGFloat, height: CGFloat) {
        if isExpanded { return (252, 40) }
        switch phase {
        case .idle:      return (56, 34)
        case .searching: return (122, 34)
        case .tracking:  return (142, 34)
        case .locked:    return (214, 46)
        }
    }

    // MARK: 岛体质感表面

    private var surface: some View {
        Capsule(style: .continuous)
            .fill(Color.black)
            .overlay(
                // 顶部极轻镜面高光
                Capsule(style: .continuous)
                    .fill(LinearGradient(colors: [.white.opacity(0.09), .clear],
                                         startPoint: .top, endPoint: .center))
                    .blendMode(.plusLighter)
            )
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.5)
            )
            .compositingGroup()
            .shadow(color: .black.opacity(0.22), radius: 10, y: 4)
    }

    // MARK: 状态内容（切换时弹性缩放 + 淡入淡出）

    @ViewBuilder
    private var content: some View {
        ZStack {
            if isExpanded {
                HStack(spacing: 8) {
                    Image(systemName: "gearshape.fill")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.green)
                    Text("功能设置")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.white)
                    Capsule()
                        .fill(Color.white.opacity(0.25))
                        .frame(width: 1, height: 14)
                    Text("点击收起")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.white.opacity(0.6))
                }
                .transition(.scale(scale: 0.7).combined(with: .opacity))
            } else {
                switch phase {
                case .idle:
                    Circle()
                        .fill(Color.gray.opacity(0.6))
                        .frame(width: 7, height: 7)
                        .transition(.scale(scale: 0.5).combined(with: .opacity))

                case .searching:
                    HStack(spacing: 7) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.white.opacity(0.7))
                        Text("搜索目标")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.white.opacity(0.85))
                    }
                    .transition(.scale(scale: 0.7).combined(with: .opacity))

                case .tracking:
                    HStack(spacing: 7) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 7, height: 7)
                            .modifier(PulseEffect(active: true))
                        Text("发现 \(targetCount) 人")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.white.opacity(0.92))
                    }
                    .transition(.scale(scale: 0.7).combined(with: .opacity))

                case .locked:
                    lockedRow
                        .transition(.scale(scale: 0.7).combined(with: .opacity))
                }
            }
        }
        .padding(.horizontal, 12)
    }

    private var lockedRow: some View {
        HStack(spacing: 9) {
            Image(systemName: "lock.fill")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.green)
            Text("已锁定 · \(targetCount) 目标")
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(.white)
            Capsule()
                .fill(Color.white.opacity(0.25))
                .frame(width: 1, height: 14)
            Text("\(fps) FPS")
                .font(.system(size: 11, weight: .semibold).monospacedDigit())
                .foregroundColor(.white.opacity(0.6))
        }
    }
}

// MARK: - 液态玻璃导航栏

/// 设置页玻璃导航栏：滚动时模糊强度、高光与阴影弹性增强（isScrolled 驱动 intensity）
struct LiquidGlassNavBar: View {

    let title: String
    let isScrolled: Bool
    let fps: Int
    let onClose: () -> Void

    var body: some View {
        ZStack {
            Text(title)
                .font(.headline)
                .foregroundColor(.primary)

            HStack {
                Button(action: onClose) {
                    Image(systemName: "chevron.down")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(.primary)
                        .frame(width: 34, height: 34)
                        .liquidGlass(cornerRadius: 17, intensity: isScrolled ? 1 : 0.5)
                }
                .buttonStyle(PressableButtonStyle())

                Spacer()

                Text("FPS \(fps)")
                    .font(.caption2.monospacedDigit().weight(.semibold))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .liquidGlass(cornerRadius: Radii.chip, intensity: isScrolled ? 1 : 0.5)
            }
        }
        .frame(height: 52)
        .overlay(alignment: .bottom) {
            // 滚动后浮现的底部发丝线（区分内容层级）
            Rectangle()
                .fill(Color.primary.opacity(0.10))
                .frame(height: 0.5)
                .opacity(isScrolled ? 1 : 0)
        }
        .padding(.horizontal, 16)
        .padding(.top, 6)
        .animation(Motion.spring, value: isScrolled)
    }
}
