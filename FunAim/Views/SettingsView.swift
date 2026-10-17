import SwiftUI

/// 功能设置面板：由灵动岛胶囊点击展开/收起（替代旧 sheet）。
/// 所有设置项与绑定关系保持一致，仅容器由整页 Form 改为横屏玻璃面板。
struct SettingsView: View {

    var onClose: (() -> Void)? = nil

    @ObservedObject private var settings = AppSettings.shared
    @ObservedObject private var camera = CameraService.shared

    /// 低饱和 Apple 系统色板：白 / 绿 / 蓝 / 橙 / 红（无紫色、无高饱和黄）
    private let colorOptions = ["#FFFFFF", "#34C759", "#007AFF", "#FF9500", "#FF3B30"]

    var body: some View {
        ZStack(alignment: .top) {
            ScrollView {
                VStack(spacing: 14) {
                    detectionCard
                    aimCard
                    displayCard
                    statusCard
                    resetButton
                }
                .padding(.horizontal, 14)
                .padding(.top, 52)
                .padding(.bottom, 22)
            }

            headerRow
        }
        .accentColor(.green)
    }

    // MARK: - 面板头（常驻，滚动内容从其下穿过；支持下拉收起）

    private var headerRow: some View {
        VStack(spacing: 4) {
            // 收起把手
            Capsule()
                .fill(Color.primary.opacity(0.25))
                .frame(width: 44, height: 5)
                .padding(.top, 6)
            HStack {
                Text("功能设置")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                if let onClose {
                    Button(action: onClose) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.primary.opacity(0.55))
                    }
                    .buttonStyle(PressableButtonStyle())
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        .frame(height: 58)
        .background(
            Rectangle()
                .fill(.ultraThinMaterial)
                .opacity(0.9)
                .frame(height: 58),
            alignment: .top
        )
        .contentShape(Rectangle())
        .highPriorityGesture(
            DragGesture()
                .onEnded { value in
                    // 下拉超过 60pt 收起面板
                    if value.translation.height > 60 {
                        onClose?()
                    }
                }
        )
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

            // 类别筛选：默认仅识别人物
            Toggle("识别全部类别（车/猫狗等）", isOn: $settings.showAllClasses)
                .font(.subheadline.weight(.medium))

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("识别灵敏度").rowTitle()
                    Spacer()
                    Text("\(Int((1 - (settings.minConfidence - 0.05) / 0.85) * 100))%")
                        .valueText()
                }
                Slider(value: $settings.minConfidence, in: 0.05...0.9)
            }

            Text("流畅每 3 帧推理一次最省电，精准逐帧推理最连贯；调高灵敏度可检出更远、更模糊的目标，但误检会增加。默认仅识别人物，开启「识别全部类别」后显示车、猫狗等所有类别并带类别标签。全部推理均在设备本地完成。")
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

            sliderRow("边框透明度", value: $settings.boxOpacity, range: 0.2...1,
                      display: "\(Int(settings.boxOpacity * 100))%")

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
