import SwiftUI

/// 叠加渲染层（ESP 风格）：人物方框 + 顶部居中射线 + 准星瞄准线 + 置信度标签
struct OverlayView: View {

    let persons: [MappedPerson]
    let reticlePoint: CGPoint
    let targetID: UUID?
    @ObservedObject private var settings = AppSettings.shared

    var body: some View {
        Canvas { context, size in
            let baseOpacity = settings.boxOpacity

            for person in persons {
                let isTarget = person.id == targetID
                let color = (isTarget ? Color.white : Color(hex: settings.boxColorHex)).opacity(baseOpacity)
                drawBox(&context, rect: person.viewRect, color: color, isTarget: isTarget)
                if settings.showConfidenceLabel {
                    drawConfidence(&context, person: person)
                }
            }

            // 顶部居中射线：从屏幕顶端中点射向每个已识别人物，锁定目标高亮
            if settings.aimEnabled, settings.showAimLine {
                let origin = CGPoint(x: size.width / 2, y: 0)
                for person in persons where person.person.isPerson {
                    drawRay(&context, from: origin, to: person.center, emphasized: person.id == targetID)
                }
            }

            if settings.aimEnabled, settings.showAimLine,
               let target = persons.first(where: { $0.id == targetID && $0.person.isPerson }) {
                drawAimLine(&context, from: reticlePoint, to: target.center)
            }

            drawCrosshair(&context, at: reticlePoint, locked: settings.aimEnabled && targetID != nil)
        }
        .animation(.linear(duration: 0.05), value: reticlePoint)
        .allowsHitTesting(false)
    }

    // MARK: - 人物方框

    private func drawBox(_ context: inout GraphicsContext, rect: CGRect, color: Color, isTarget: Bool) {
        guard let path = boxPath(rect, style: settings.boxStyle) else { return }
        let width = CGFloat(settings.boxThickness)
        let stroke = StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round)
        if isTarget {
            // 锁定目标：外圈辉光 + 白色高亮描边
            context.stroke(path, with: .color(color.opacity(0.30)), lineWidth: width + 6)
        }
        context.stroke(path, with: .color(color.opacity(0.95)), style: stroke)
    }

    private func boxPath(_ rect: CGRect, style: BoxStyle) -> Path? {
        guard rect.width > 1, rect.height > 1 else { return nil }
        var path = Path()
        switch style {
        case .fullRect:
            path.addRoundedRect(in: rect, cornerSize: CGSize(width: 8, height: 8), style: .continuous)
        case .brackets:
            let len = min(CGFloat(18), rect.width / 2.5, rect.height / 2.5)
            // 左上
            path.move(to: CGPoint(x: rect.minX, y: rect.minY + len))
            path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
            path.addLine(to: CGPoint(x: rect.minX + len, y: rect.minY))
            // 右上
            path.move(to: CGPoint(x: rect.maxX - len, y: rect.minY))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + len))
            // 右下
            path.move(to: CGPoint(x: rect.maxX, y: rect.maxY - len))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
            path.addLine(to: CGPoint(x: rect.maxX - len, y: rect.maxY))
            // 左下
            path.move(to: CGPoint(x: rect.minX + len, y: rect.maxY))
            path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
            path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - len))
        }
        return path
    }

    private func drawConfidence(_ context: inout GraphicsContext, person: MappedPerson) {
        var parts: [String] = []
        if person.person.isPerson {
            if let state = person.person.state, state != .unknown {
                parts.append(state.rawValue)
            }
            if let face = person.person.faceVisible {
                parts.append(face ? "露脸" : "未露脸")
            }
        } else {
            parts.append(person.person.label)
            if let cm = person.person.estimatedHeightCm {
                parts.append("≈\(Int(cm))cm")
            }
        }
        parts.append("\(Int(person.person.confidence * 100))%")
        let text = Text(parts.joined(separator: " · "))
            .font(.system(size: 11, weight: .bold))
            .foregroundColor(.white)
        let resolved = context.resolve(text)
        let textSize = resolved.measure(in: CGSize(width: 100, height: 20))
        let centerY = max(person.viewRect.minY - 11, 12)
        let centerX = person.viewRect.midX
        let bgRect = CGRect(x: centerX - textSize.width / 2 - 4,
                            y: centerY - textSize.height / 2 - 2,
                            width: textSize.width + 8,
                            height: textSize.height + 4)
        context.fill(Path(roundedRect: bgRect, cornerRadius: 6), with: .color(.black.opacity(0.45)))
        context.draw(resolved, at: CGPoint(x: centerX, y: centerY))
    }

    // MARK: - 顶部居中射线（ESP snapline）

    private func drawRay(_ context: inout GraphicsContext, from origin: CGPoint, to point: CGPoint, emphasized: Bool) {
        var path = Path()
        path.move(to: origin)
        path.addLine(to: point)
        if emphasized {
            // 锁定目标：辉光 + 高亮
            context.stroke(path, with: .color(.white.opacity(0.15)), lineWidth: 6)
        }
        let color = emphasized ? Color.white.opacity(0.95) : Color.white.opacity(0.30)
        context.stroke(path, with: .color(color),
                       style: StrokeStyle(lineWidth: emphasized ? 2.5 : 1.5, lineCap: .round))

        // 目标端点标记
        var dot = Path()
        let radius: CGFloat = emphasized ? 5 : 3
        dot.addEllipse(in: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2))
        context.fill(dot, with: .color(emphasized ? .green : .white.opacity(0.6)))
    }

    // MARK: - 准星瞄准线

    private func drawAimLine(_ context: inout GraphicsContext, from: CGPoint, to: CGPoint) {
        var path = Path()
        path.move(to: from)
        path.addLine(to: to)
        let style: StrokeStyle = settings.aimLineStyle == .dashed
            ? StrokeStyle(lineWidth: 2.5, lineCap: .round, dash: [10, 7])
            : StrokeStyle(lineWidth: 2.5, lineCap: .round)
        context.stroke(path, with: .color(.white.opacity(0.85)), style: style)

        // 箭头样式：目标端绘制箭头
        if settings.aimLineStyle == .arrow {
            let angle = atan2(to.y - from.y, to.x - from.x)
            let len = CGFloat(14)
            let a = CGPoint(x: to.x - len * cos(angle - CGFloat.pi / 7),
                            y: to.y - len * sin(angle - CGFloat.pi / 7))
            let b = CGPoint(x: to.x - len * cos(angle + CGFloat.pi / 7),
                            y: to.y - len * sin(angle + CGFloat.pi / 7))
            var head = Path()
            head.move(to: to)
            head.addLine(to: a)
            head.addLine(to: b)
            head.closeSubpath()
            context.fill(head, with: .color(.white.opacity(0.9)))
        }

        // 目标中心点（绿色 = 系统强调色，与锁定态语义一致）
        var dot = Path()
        dot.addEllipse(in: CGRect(x: to.x - 3.5, y: to.y - 3.5, width: 7, height: 7))
        context.fill(dot, with: .color(.green))
    }

    // MARK: - 准星

    private func drawCrosshair(_ context: inout GraphicsContext, at point: CGPoint, locked: Bool) {
        let color: Color = locked ? .green : .white

        var ring = Path()
        ring.addEllipse(in: CGRect(x: point.x - 13, y: point.y - 13, width: 26, height: 26))
        context.stroke(ring, with: .color(color.opacity(0.9)), lineWidth: 2)

        var dot = Path()
        dot.addEllipse(in: CGRect(x: point.x - 2, y: point.y - 2, width: 4, height: 4))
        context.fill(dot, with: .color(color))

        let ticks: [(CGFloat, CGFloat)] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for (dx, dy) in ticks {
            var tick = Path()
            tick.move(to: CGPoint(x: point.x + dx * 16, y: point.y + dy * 16))
            tick.addLine(to: CGPoint(x: point.x + dx * 24, y: point.y + dy * 24))
            context.stroke(tick, with: .color(color.opacity(0.7)), lineWidth: 2)
        }
    }
}
