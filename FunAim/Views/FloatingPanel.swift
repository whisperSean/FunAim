import SwiftUI

/// 悬浮控制窗：液态玻璃材质的可拖拽面板，弹性动效反馈
struct FloatingPanel: View {

    @ObservedObject var camera: CameraService
    @Binding var aimEnabled: Bool
    var onOpenSettings: () -> Void

    @State private var dragOffset: CGSize = .zero
    @State private var dragStart: CGSize = .zero

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 6) {
                Circle()
                    .fill(aimEnabled ? Color.green : Color.gray)
                    .frame(width: 8, height: 8)
                    .modifier(PulseEffect(active: aimEnabled))
                Text(aimEnabled ? "自动瞄准已开启" : "自动瞄准已关闭")
                    .font(.caption.bold())
                    .foregroundColor(.primary)
                Spacer(minLength: 0)
                Image(systemName: "line.3.horizontal")
                    .font(.caption2.weight(.semibold))
                    .foregroundColor(.secondary)
            }
            .animation(Motion.spring, value: aimEnabled)

            Text("FPS \(Int(camera.fps)) · \(String(format: "%.0f", camera.inferenceMs))ms · 目标 \(camera.detectionCount)")
                .font(.caption2.monospacedDigit())
                .foregroundColor(.secondary)

            Toggle("瞄准", isOn: $aimEnabled)
                .toggleStyle(.switch)
                .tint(.green)
                .font(.caption)

            Button(action: onOpenSettings) {
                Label("设置", systemImage: "gearshape.fill")
                    .font(.caption.bold())
                    .foregroundColor(.primary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 7)
                    .background(
                        RoundedRectangle(cornerRadius: Radii.chip, style: .continuous)
                            .fill(Color.primary.opacity(0.07))
                    )
            }
            .buttonStyle(PressableButtonStyle())
        }
        .padding(14)
        .frame(width: 196)
        .liquidGlass(cornerRadius: Radii.card, intensity: 0.95)
        .offset(dragOffset)
        .gesture(dragGesture)
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                dragOffset = CGSize(width: dragStart.width + value.translation.width,
                                    height: dragStart.height + value.translation.height)
            }
            .onEnded { _ in dragStart = dragOffset }
    }
}
