import XCTest
@testable import FunAim

// MARK: - 坐标映射测试

final class CoordinateMapperTests: XCTestCase {

    func testCenterStaysAtCenter() {
        let mapper = CoordinateMapper(viewSize: CGSize(width: 800, height: 400),
                                      bufferSize: CGSize(width: 1280, height: 720))
        let mapped = mapper.map(CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5))
        XCTAssertEqual(mapped.midX, 400, accuracy: 0.5)
        XCTAssertEqual(mapped.midY, 200, accuracy: 0.5)
    }

    func testFullFrameCoversViewWithAspectFill() {
        let mapper = CoordinateMapper(viewSize: CGSize(width: 800, height: 400),
                                      bufferSize: CGSize(width: 1280, height: 720))
        let mapped = mapper.map(CGRect(x: 0, y: 0, width: 1, height: 1))
        // 等比填充：整幅画面映射到"缩放后缓冲区尺寸"——
        // 限制维度（宽度 800）精确撑满，另一维度（高度 450）按比例溢出，溢出部分被屏幕裁剪
        XCTAssertEqual(mapped.width, 800, accuracy: 1.0)
        XCTAssertEqual(mapped.height, 450, accuracy: 1.0)
        XCTAssertGreaterThanOrEqual(mapped.height, 400)
    }

    func testTopOfBufferMapsToTopOfView() {
        let mapper = CoordinateMapper(viewSize: CGSize(width: 800, height: 400),
                                      bufferSize: CGSize(width: 1280, height: 720))
        // Vision 坐标 y 向上：maxY=1 表示画面顶部
        let mapped = mapper.map(CGRect(x: 0.4, y: 0.9, width: 0.2, height: 0.1))
        XCTAssertLessThan(mapped.midY, 100)
    }

    func testZeroSizeReturnsZero() {
        let mapper = CoordinateMapper(viewSize: .zero, bufferSize: CGSize(width: 1280, height: 720))
        XCTAssertEqual(mapper.map(CGRect(x: 0, y: 0, width: 0.5, height: 0.5)), .zero)
    }
}

// MARK: - 姿态分类测试

final class PoseClassifierTests: XCTestCase {

    private func keypoints(noseVisible: Bool = true,
                           hipsVisible: Bool = true,
                           kneesVisible: Bool = true,
                           anklesVisible: Bool = true,
                           legLength: Double = 0.4) -> [Keypoint] {
        var kp = [Keypoint](repeating: Keypoint(x: 0.5, y: 0.5, visible: false), count: 17)
        kp[KeypointIndex.nose] = Keypoint(x: 0.5, y: 0.1, visible: noseVisible)
        kp[KeypointIndex.leftShoulder] = Keypoint(x: 0.4, y: 0.3, visible: true)
        kp[KeypointIndex.rightShoulder] = Keypoint(x: 0.6, y: 0.3, visible: true)
        kp[KeypointIndex.leftHip] = Keypoint(x: 0.45, y: 0.5, visible: hipsVisible)
        kp[KeypointIndex.rightHip] = Keypoint(x: 0.55, y: 0.5, visible: hipsVisible)
        kp[KeypointIndex.leftKnee] = Keypoint(x: 0.45, y: 0.5 + legLength / 2, visible: kneesVisible)
        kp[KeypointIndex.rightKnee] = Keypoint(x: 0.55, y: 0.5 + legLength / 2, visible: kneesVisible)
        kp[KeypointIndex.leftAnkle] = Keypoint(x: 0.45, y: 0.5 + legLength, visible: anklesVisible)
        kp[KeypointIndex.rightAnkle] = Keypoint(x: 0.55, y: 0.5 + legLength, visible: anklesVisible)
        return kp
    }

    func testStanding() {
        let (state, face) = PoseClassifier.classify(keypoints: keypoints(legLength: 0.4))
        XCTAssertEqual(state, .standing)
        XCTAssertTrue(face)
    }

    func testSquatting() {
        // 蹲姿：小腿被压缩（躯干 0.2，腿 0.1 → 比值 0.5 < 0.6）
        let (state, _) = PoseClassifier.classify(keypoints: keypoints(legLength: 0.1))
        XCTAssertEqual(state, .squatting)
    }

    func testHalfBodyWhenHipsMissing() {
        let (state, face) = PoseClassifier.classify(keypoints: keypoints(hipsVisible: false, noseVisible: true))
        XCTAssertEqual(state, .halfBody)
        XCTAssertTrue(face)
    }

    func testNoFaceWhenNoseHidden() {
        let (_, face) = PoseClassifier.classify(keypoints: keypoints(noseVisible: false))
        XCTAssertFalse(face)
    }

    func testUnknownWhenLowerBodyIncomplete() {
        let (state, _) = PoseClassifier.classify(keypoints: keypoints(anklesVisible: false))
        XCTAssertEqual(state, .unknown)
    }
}

// MARK: - 尺寸估算测试

final class SizeEstimatorTests: XCTestCase {

    func testEstimateWithPersonReference() {
        // 人物框高 0.5（=170cm），物体框高 0.25 → 85cm
        let est = SizeEstimator.estimate(objectBox: CGRect(x: 0, y: 0, width: 0.2, height: 0.25),
                                         personBox: CGRect(x: 0, y: 0, width: 0.3, height: 0.5))
        XCTAssertEqual(est.estimatedHeightCm ?? 0, 85, accuracy: 0.1)
        XCTAssertEqual(est.relativeHeight, 0.25, accuracy: 0.001)
    }

    func testNoPersonReturnsRelativeOnly() {
        let est = SizeEstimator.estimate(objectBox: CGRect(x: 0, y: 0, width: 0.2, height: 0.25),
                                         personBox: nil)
        XCTAssertNil(est.estimatedHeightCm)
        XCTAssertEqual(est.relativeHeight, 0.25, accuracy: 0.001)
    }

    func testTinyPersonReferenceIgnored() {
        // 参照人物框过小（<0.02）视为无效
        let est = SizeEstimator.estimate(objectBox: CGRect(x: 0, y: 0, width: 0.2, height: 0.25),
                                         personBox: CGRect(x: 0, y: 0, width: 0.01, height: 0.01))
        XCTAssertNil(est.estimatedHeightCm)
    }
}

// MARK: - 跟踪器测试

final class PersonTrackerTests: XCTestCase {

    private func target(_ x: Double, _ y: Double, conf: Float = 0.9,
                        label: String = "person", isPerson: Bool = true) -> RawTarget {
        RawTarget(box: CGRect(x: x, y: y, width: 0.2, height: 0.4),
                  conf: conf, label: label, isPerson: isPerson,
                  state: isPerson ? .standing : .unknown,
                  faceVisible: isPerson, estimatedHeightCm: nil)
    }

    func testStableIDAcrossFrames() {
        let tracker = PersonTracker()
        let first = tracker.assign([target(0.3, 0.3)])
        let second = tracker.assign([target(0.32, 0.32)])
        XCTAssertEqual(first.count, 1)
        XCTAssertEqual(second.count, 1)
        XCTAssertEqual(first[0].id, second[0].id)
    }

    func testDifferentTargetsGetDifferentIDs() {
        let tracker = PersonTracker()
        let first = tracker.assign([target(0.3, 0.3)])
        let second = tracker.assign([target(0.9, 0.9)])   // 远离，视为新目标
        XCTAssertNotEqual(first[0].id, second[0].id)
    }

    func testCoastOneFrameThenDisappear() {
        let tracker = PersonTracker()
        _ = tracker.assign([target(0.3, 0.3)])
        let coasted = tracker.assign([])     // 遮挡 1 帧：滑行输出
        XCTAssertEqual(coasted.count, 1)
        let gone = tracker.assign([])        // 超限：移除
        XCTAssertEqual(gone.count, 0)
    }

    func testItemsCarryLabelAndNotPerson() {
        let tracker = PersonTracker()
        let result = tracker.assign([target(0.3, 0.3, label: "cup", isPerson: false)])
        XCTAssertEqual(result[0].label, "cup")
        XCTAssertFalse(result[0].isPerson)
        XCTAssertNil(result[0].state)
        XCTAssertNil(result[0].estimatedHeightCm)
    }

    func testSmoothedBoxConvergesToMeasurement() {
        let tracker = PersonTracker()
        _ = tracker.assign([target(0.0, 0.0)])
        // 连续多帧后 EMA 收敛到测量位置
        var last: DetectedPerson?
        for _ in 0..<30 {
            last = tracker.assign([target(0.5, 0.5)]).first
        }
        XCTAssertEqual(last?.box.minX ?? 0, 0.5, accuracy: 0.02)
    }
}

// MARK: - 张量解码测试

final class DecodeTests: XCTestCase {

    /// 构造 [1, C, N] 张量并在指定候选位写入数据
    private func buildTensor(channels: Int, count: Int,
                             write: (Int, Int, (Float) -> Void) -> Void) -> MLMultiArray {
        let mla = MLMultiArray(shape: [1, channels, count] as [NSNumber], dataType: .float32)
        write(channels, count, { channel, index, value in
            mla[channel * count + index] = NSNumber(value: value)
        })
        return mla
    }

    func testDecodeItemsWhitelistAndPersonExcluded() throws {
        let count = 8400
        var tensor: MLMultiArray?
        tensor = buildTensor(channels: 84, count: count) { channels, _, put in
            // 候选 100：person（class 0）
            put(0, 100, 320); put(1, 100, 320); put(2, 100, 60); put(3, 100, 160)
            put(4, 100, 0.95)
            // 候选 200：cup（COCO class 41 → 通道 45）
            put(0, 200, 320); put(1, 200, 400); put(2, 200, 80); put(3, 200, 120)
            put(45, 200, 0.9)
            // 候选 300：car（class 2 → 通道 6，不在白名单）
            put(0, 300, 320); put(1, 300, 400); put(2, 300, 120); put(3, 300, 80)
            put(6, 300, 0.93)
        }
        let result = PersonDetector.decodeItems(tensor!, includePersons: false, minConfidence: 0.5)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].label, "cup")
        XCTAssertFalse(result[0].isPerson)
    }

    func testDecodeItemsIncludesPersonWhenRequested() throws {
        let count = 8400
        let tensor = buildTensor(channels: 84, count: count) { _, _, put in
            put(0, 100, 320); put(1, 100, 320); put(2, 100, 60); put(3, 100, 160)
            put(4, 100, 0.95)
        }
        let result = PersonDetector.decodeItems(tensor, includePersons: true, minConfidence: 0.5)
        XCTAssertEqual(result.count, 1)
        XCTAssertTrue(result[0].isPerson)
    }

    func testDecodePoseKeypoints() throws {
        let count = 8400
        let tensor = buildTensor(channels: 56, count: count) { _, _, put in
            put(0, 500, 320); put(1, 500, 320); put(2, 500, 100); put(3, 500, 300)
            put(4, 500, 0.95)                                   // 置信度
            put(5, 500, 320); put(6, 500, 200); put(7, 500, 0.9) // 鼻子 x/y/vis
        }
        let result = PersonDetector.decodePose(tensor, minConfidence: 0.5)
        XCTAssertEqual(result.count, 1)
        XCTAssertTrue(result[0].isPerson)
        XCTAssertTrue(result[0].faceVisible == true)   // 鼻子可见 → 露脸
        XCTAssertEqual(result[0].state, .unknown)      // 其余关键点不可见 → 无法判定站蹲
    }
}
