import CoreGraphics
import Foundation

// MARK: - 检测结果类型

/// 人物状态（由关键点几何推导）
enum PersonState: String {
    case standing = "站立"
    case squatting = "蹲姿"
    case halfBody = "半身"
    case unknown = "未知"
}

/// 单个检测结果（人物或生活物品）
/// box 为归一化坐标（原点左下，与 Vision/CoreML 一致）
struct DetectedPerson {
    let id: UUID
    let box: CGRect
    let confidence: Float
    let label: String
    let isPerson: Bool
    /// 人物姿态（仅人物有值）
    let state: PersonState?
    /// 人脸是否露出（仅人物有值）
    let faceVisible: Bool?
    /// 物品估算实际高度（cm，参照画面中人物身高 1.7m；仅物品且同帧有人物时有值）
    let estimatedHeightCm: Double?
    /// 瞄准点（归一化 Vision 坐标）：人物=头部（鼻/肩上推），物品=中心
    let aimPoint: CGPoint?
}

// MARK: - 姿态分类器（纯逻辑，可单测）

/// COCO 17 关键点索引
enum KeypointIndex {
    static let nose = 0
    static let leftEye = 1, rightEye = 2
    static let leftShoulder = 5, rightShoulder = 6
    static let leftHip = 11, rightHip = 12
    static let leftKnee = 13, rightKnee = 14
    static let leftAnkle = 15, rightAnkle = 16
}

struct Keypoint {
    let x: Double       // 归一化（0~1，相对模型输入）
    let y: Double
    let visible: Bool
}

/// 从 17 关键点推导人物状态与人脸露出情况
enum PoseClassifier {

    static let visibilityThreshold = 0.3

    static func classify(keypoints: [Keypoint]) -> (state: PersonState, faceVisible: Bool) {
        guard keypoints.count == 17 else { return (.unknown, false) }

        // 人脸露出：鼻子可见（眼部遮挡时鼻子通常仍可见，鲁棒性更好）
        let faceVisible = keypoints[KeypointIndex.nose].visible

        let leftHip = keypoints[KeypointIndex.leftHip]
        let rightHip = keypoints[KeypointIndex.rightHip]
        let hipsVisible = leftHip.visible || rightHip.visible

        // 半身：髋部不可见（下半身不在画面内）
        guard hipsVisible else {
            let shouldersVisible = keypoints[KeypointIndex.leftShoulder].visible
                || keypoints[KeypointIndex.rightShoulder].visible
            return (shouldersVisible ? .halfBody : .unknown, faceVisible)
        }

        let leftKnee = keypoints[KeypointIndex.leftKnee]
        let rightKnee = keypoints[KeypointIndex.rightKnee]
        let leftAnkle = keypoints[KeypointIndex.leftAnkle]
        let rightAnkle = keypoints[KeypointIndex.rightAnkle]
        let kneesVisible = leftKnee.visible || rightKnee.visible
        let anklesVisible = leftAnkle.visible || rightAnkle.visible

        // 下半身不完整无法判定站蹲
        guard kneesVisible, anklesVisible else { return (.unknown, faceVisible) }

        // 几何判定：小腿长度（髋中点-踝中点）与躯干长度（肩中点-髋中点）之比
        // 站立 ≈ 1.0+；蹲姿明显压缩 < 0.6
        let hipMid = mid(leftHip, rightHip)
        let ankleMid = mid(leftAnkle, rightAnkle)
        let shoulderMid = mid(keypoints[KeypointIndex.leftShoulder], keypoints[KeypointIndex.rightShoulder])
        let legLength = hypot(ankleMid.x - hipMid.x, ankleMid.y - hipMid.y)
        let torsoLength = max(hypot(shoulderMid.x - hipMid.x, shoulderMid.y - hipMid.y), 0.001)

        return (legLength / torsoLength < 0.6 ? .squatting : .standing, faceVisible)
    }

    private static func mid(_ a: Keypoint, _ b: Keypoint) -> CGPoint {
        if a.visible && b.visible {
            return CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
        }
        return a.visible ? CGPoint(x: a.x, y: a.y) : CGPoint(x: b.x, y: b.y)
    }
}

// MARK: - 尺寸估算器（纯逻辑，可单测）

struct SizeEstimate {
    /// 物体框高占画面高度的比例（0~1）
    let relativeHeight: Double
    /// 估算实际高度（cm）。参照物：同帧人物（假设身高 170cm、与物体同深度平面）
    /// 无人物参照时为 nil —— 单目摄像头无距离信息，不编造数字
    let estimatedHeightCm: Double?
}

enum SizeEstimator {
    static let referencePersonHeightCm = 170.0

    /// objectBox/personBox 为同一坐标系下的归一化框
    static func estimate(objectBox: CGRect, personBox: CGRect?) -> SizeEstimate {
        let relative = Double(objectBox.height)
        guard let personBox, personBox.height > 0.02 else {
            return SizeEstimate(relativeHeight: relative, estimatedHeightCm: nil)
        }
        // 同深度平面假设：物体高度 / 人物框高 × 170cm
        let realCm = relative / Double(personBox.height) * referencePersonHeightCm
        return SizeEstimate(relativeHeight: relative, estimatedHeightCm: realCm)
    }
}
