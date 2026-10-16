import SwiftUI

/// 功能设置：滚动响应液态玻璃导航栏 + 玻璃卡片分组
/// 视觉遵循 iOS 原生规范：continuous 大圆角、系统语义色、弹性动效曲线；
/// 所有设置项与绑定关系与旧版完全一致，仅重构视觉表现。
struct SettingsView: View {

    @ObservedObject private var settings = AppSettings.shared
    @ObservedObject private var camera = CameraService.shared
    @Environment(\.dismiss) private var dismiss

    /// 低饱和 Apple 系统色板：白 / 绿 / 蓝 / 橙 / 红（无紫色、无高饱和黄）
    private let colorOptions = ["#FFFFFF", "#34C759", "#007AFF", "#FF9500", "#FF3B30"]

    /// 滚动超过阈值后导航栏玻璃强度弹性增强（状态翻转才写入，避免滚动逐帧触发重算）
    @State private var isScrolled = false

    var body: some View {
        ZStack(alignment: .top) {
            Color(uiColor: .systemGroupedBackground)
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 14) {
                    detectionCard
                    aimCard
                    displayCard
                    statusCard
                    resetButton
                }
                .padding(.horizontal, 14)
                .padding(.top, 70)
                .padding(.bottom, 28)
                .background(scrollOffsetReader)
            }
            .coordinateSpace(name: "settings.scroll")
            .onPreferenceChange(ScrollOffsetKey.self) { offset in
                let newValue = offset > 8
                if newValue != isScrolled { isScrolled = newValue }
            }

            LiquidGlassNavBar(title: "功能设置",
                              isScrolled: isScrolled,
                              fps: Int(camera.fps)) {
                dismiss()
            }
        }
        .accentColor(.green)
    }

    // MARK: - 滚动监听

    private var scrollOffsetReader: some View {
        GeometryReader { geo in
            Color.clear.preference(
                key: ScrollOffsetKey.self,
                value: -geo.frame(in: .named("settings.scroll")).minY
            )
        }
    }

    // MARK: - 检测模型

    private var detectionCard: some View {
        glassCard {
            cardHeader("cpu.fill", "检测模型")

            VStack(alignment: .leading, spacing: 8) {
                Text("运行模式").rowTitle()
                Picker("", selection: $settings.detectionMode) {
                    ForEach(DetectionMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("识别灵敏度").rowTitle()
                    Spacer()
                    Text("\(Int((1 - (settings.minConfidence - 0.05) / 0.85) * 100))%")
                        .valueText()
                }
                Slider(value: $settings.minConfidence, in: 0.05...0.9)
            }

            Text("流畅每 3 帧推理一次最省电，精准逐帧推理最连贯；调高灵敏度可检出更远、更模糊的人物，但误检会增加。全部推理均在设备本地完成。")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - 自动瞄准

    private var aimCard: some View {
        glassCard {
            cardHeader("scope", "自动瞄准")

            Toggle("瞄准总开关", isOn: $settings.aimEnabled)
                .font(.subheadline.weight(.medium))

            // 关闭瞄准时选项降透明度并禁用，卡片高度不变，过渡平滑
            Group {
                sliderRow("跟踪速度", value: $settings.aimTracking, range: 0.05...1)
                sliderRow("锁定强度", value: $settings.lockStrength, range: 0...1)
                toggleRow("显示瞄准射线", $settings.showAimLine)

                VStack(alignment: .leading, spacing: 8) {
                    Text("射线样式").rowTitle()
                    Picker("", selection: $settings.aimLineStyle) {
                        ForEach(AimLineStyle.allCases) { style in
                            Text(style.rawValue).tag(style)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
            }
            .opacity(settings.aimEnabled ? 1 : 0.35)
            .disabled(!settings.aimEnabled)

            Text("跟踪速度决定准星贴合目标的速度；锁定强度决定画面中出现多个人物时，准星保持当前目标的倾向。")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .animation(Motion.spring, value: settings.aimEnabled)
    }

    // MARK: - 显示样式

    private var displayCard: some View {
        glassCard {
            cardHeader("paintbrush.fill", "显示样式")

            VStack(alignment: .leading, spacing: 8) {
                Text("边框样式").rowTitle()
                Picker("", selection: $settings.boxStyle) {
                    ForEach(BoxStyle.allCases) { style in
                        Text(style.rawValue).tag(style)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            sliderRow("边框粗细", value: $settings.boxThickness, range: 1...8,
                      display: String(format: "%.0f", settings.boxThickness))

            VStack(alignment: .leading, spacing: 10) {
                Text("标注颜色").rowTitle()
                HStack(spacing: 14) {
                    ForEach(colorOptions, id: \.self) { hex in
                        swatch(hex: hex)
                    }
                    Spacer()
                }
            }

            toggleRow("显示置信度标签", $settings.showConfidenceLabel)
            toggleRow("显示悬浮控制窗", $settings.showFloatPanel)
            toggleRow("显示性能 HUD", $settings.showHUD)
        }
    }

    // MARK: - 运行状态

    private var statusCard: some View {
        glassCard {
            cardHeader("gauge", "运行状态")
            statusRow("预览帧率", "\(Int(camera.fps)) FPS")
            statusRow("单帧推理", String(format: "%.1f ms", camera.inferenceMs))
            statusRow("当前目标数", "\(camera.detectionCount)")
            statusRow("推理引擎", "Vision · 设备本地")
        }
    }

    // MARK: - 恢复默认

    private var resetButton: some View {
        Button {
            withAnimation(Motion.spring) { settings.resetToDefaults() }
        } label: {
            Text("恢复默认设置")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.red)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 13)
        }
        .liquidGlass(cornerRadius: Radii.card, intensity: 0.6)
        .buttonStyle(PressableButtonStyle())
    }

    // MARK: - 组件

    private func glassCard<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 14) { content() }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .liquidGlass(cornerRadius: Radii.card, intensity: 0.85)
    }

    private func cardHeader(_ icon: String, _ title: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.subheadline.weight(.bold))
                .foregroundColor(.accentColor)
                .frame(width: 26, height: 26)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.accentColor.opacity(0.14))
                )
            Text(title)
                .font(.subheadline.weight(.bold))
                .foregroundColor(.primary)
        }
    }

    private func sliderRow(_ title: String,
                           value: Binding<Double>,
                           range: ClosedRange<Double>,
                           display: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).rowTitle()
                Spacer()
                Text(display ?? "\(Int(value.wrappedValue * 100))%")
                    .valueText()
            }
            Slider(value: value, in: range)
        }
    }

    private func toggleRow(_ title: String, _ binding: Binding<Bool>) -> some View {
        HStack {
            Text(title)
                .font(.subheadline.weight(.medium))
            Spacer()
            Toggle("", isOn: binding)
                .labelsHidden()
                .tint(.green)
        }
    }

    private func statusRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
                .font(.subheadline.weight(.medium))
            Spacer()
            Text(value).valueText()
        }
    }

    private func swatch(hex: String) -> some View {
        let selected = settings.boxColorHex == hex
        return Circle()
            .fill(Color(hex: hex))
            .frame(width: 30, height: 30)
            .overlay(
                // 浅色色板在卡片上的可见性发丝描边
                Circle()
                    .strokeBorder(Color.primary.opacity(0.10), lineWidth: 1)
            )
            .overlay(
                // 选中态：外圈选中环 + 弹性放大，不做多余装饰
                Circle()
                    .stroke(Color.primary.opacity(selected ? 0.85 : 0), lineWidth: 2)
                    .padding(-4)
            )
            .scaleEffect(selected ? 1.12 : 1)
            .animation(Motion.springSnappy, value: selected)
            .onTapGesture { settings.boxColorHex = hex }
    }
}

// MARK: - 文本样式助手

private extension Text {
    func rowTitle() -> Text {
        font(.subheadline.weight(.medium)).foregroundColor(.primary)
    }
    func valueText() -> Text {
        font(.subheadline.monospacedDigit()).foregroundColor(.secondary)
    }
}

// MARK: - 滚动偏移

private struct ScrollOffsetKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}
