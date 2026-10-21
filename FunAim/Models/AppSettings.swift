import SwiftUI
import Foundation

// MARK: - 配置枚举

/// 运行模式：控制 AI 推理帧间隔（预览画面始终满帧渲染）
enum DetectionMode: String, CaseIterable, Identifiable {
    case fast = "流畅"
    case balanced = "均衡"
    case accurate = "精准"

    var id: String { rawValue }
    var frameStride: Int {
        switch self {
        case .fast: return 3
        case .balanced: return 2
        case .accurate: return 1
        }
    }
}

enum BoxStyle: String, CaseIterable, Identifiable {
    case brackets = "角框"
    case fullRect = "全框"
    var id: String { rawValue }
}

enum AimLineStyle: String, CaseIterable, Identifiable {
    case solid = "实线"
    case dashed = "虚线"
    case arrow = "箭头"
    var id: String { rawValue }
}

// MARK: - 设置中心（内存运行 + UserDefaults 持久化）

final class AppSettings: ObservableObject {

    static let shared = AppSettings()
    private let d = UserDefaults.standard

    // 检测模型
    @Published var detectionMode: DetectionMode { didSet { d.set(detectionMode.rawValue, forKey: "fa.detectionMode") } }
    /// 最低置信度阈值：越低越灵敏
    @Published var minConfidence: Double { didSet { d.set(minConfidence, forKey: "fa.minConfidence") } }
    /// 人物姿态识别（站立/蹲姿/半身 + 人脸露出），默认开启
    @Published var poseEnabled: Bool { didSet { d.set(poseEnabled, forKey: "fa.poseEnabled") } }
    /// 生活物品识别（32 类家居白名单），默认开启
    @Published var itemsEnabled: Bool { didSet { d.set(itemsEnabled, forKey: "fa.itemsEnabled") } }
    /// 画面旋转 180°：方向与实际相反时手动校正
    @Published var flipRotation: Bool { didSet { d.set(flipRotation, forKey: "fa.flipRotation") } }

    // 自动瞄准
    @Published var aimEnabled: Bool { didSet { d.set(aimEnabled, forKey: "fa.aimEnabled") } }
    /// 跟踪速度：准星贴合目标的快慢
    @Published var aimTracking: Double { didSet { d.set(aimTracking, forKey: "fa.aimTracking") } }
    /// 锁定强度：多目标时保持当前目标的粘滞度
    @Published var lockStrength: Double { didSet { d.set(lockStrength, forKey: "fa.lockStrength") } }
    @Published var showAimLine: Bool { didSet { d.set(showAimLine, forKey: "fa.showAimLine") } }
    @Published var aimLineStyle: AimLineStyle { didSet { d.set(aimLineStyle.rawValue, forKey: "fa.aimLineStyle") } }

    // 显示样式
    @Published var boxStyle: BoxStyle { didSet { d.set(boxStyle.rawValue, forKey: "fa.boxStyle") } }
    @Published var boxThickness: Double { didSet { d.set(boxThickness, forKey: "fa.boxThickness") } }
    @Published var boxOpacity: Double { didSet { d.set(boxOpacity, forKey: "fa.boxOpacity") } }
    @Published var boxColorHex: String { didSet { d.set(boxColorHex, forKey: "fa.boxColorHex") } }
    @Published var showConfidenceLabel: Bool { didSet { d.set(showConfidenceLabel, forKey: "fa.showConfidenceLabel") } }
    @Published var showHUD: Bool { didSet { d.set(showHUD, forKey: "fa.showHUD") } }

    private init() {
        detectionMode = DetectionMode(rawValue: d.string(forKey: "fa.detectionMode") ?? "") ?? .balanced
        minConfidence = d.object(forKey: "fa.minConfidence") as? Double ?? 0.65
        poseEnabled = d.object(forKey: "fa.poseEnabled") as? Bool ?? true
        itemsEnabled = d.object(forKey: "fa.itemsEnabled") as? Bool ?? true
        flipRotation = d.object(forKey: "fa.flipRotation") as? Bool ?? false
        aimEnabled = d.object(forKey: "fa.aimEnabled") as? Bool ?? true
        aimTracking = d.object(forKey: "fa.aimTracking") as? Double ?? 0.35
        lockStrength = d.object(forKey: "fa.lockStrength") as? Double ?? 0.6
        showAimLine = d.object(forKey: "fa.showAimLine") as? Bool ?? true
        aimLineStyle = AimLineStyle(rawValue: d.string(forKey: "fa.aimLineStyle") ?? "") ?? .arrow
        boxStyle = BoxStyle(rawValue: d.string(forKey: "fa.boxStyle") ?? "") ?? .brackets
        boxThickness = d.object(forKey: "fa.boxThickness") as? Double ?? 3
        boxOpacity = d.object(forKey: "fa.boxOpacity") as? Double ?? 0.9
        boxColorHex = d.string(forKey: "fa.boxColorHex") ?? "#34C759"
        showConfidenceLabel = d.object(forKey: "fa.showConfidenceLabel") as? Bool ?? true
        showHUD = d.object(forKey: "fa.showHUD") as? Bool ?? true
    }

    func resetToDefaults() {
        detectionMode = .balanced
        minConfidence = 0.65
        poseEnabled = true
        itemsEnabled = true
        flipRotation = false
        aimEnabled = true
        aimTracking = 0.35
        lockStrength = 0.6
        showAimLine = true
        aimLineStyle = .arrow
        boxStyle = .brackets
        boxThickness = 3
        boxOpacity = 0.9
        boxColorHex = "#34C759"
        showConfidenceLabel = true
        showHUD = true
    }
}

// MARK: - 颜色工具

extension Color {
    init(hex: String) {
        var s = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("#") { s.removeFirst() }
        var value: UInt64 = 0
        Scanner(string: s).scanHexInt64(&value)
        let r, g, b: Double
        if s.count == 6 {
            r = Double((value >> 16) & 0xFF) / 255
            g = Double((value >> 8) & 0xFF) / 255
            b = Double(value & 0xFF) / 255
        } else {
            r = 1; g = 1; b = 1
        }
        self.init(.sRGB, red: r, green: g, blue: b, opacity: 1)
    }
}
