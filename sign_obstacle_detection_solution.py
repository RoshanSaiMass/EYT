'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2B of Niti Vahan (NV) Theme.
*
*  Filename:         sign_obstacle_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import cv2
import numpy as np
##############################################################

# The only sign labels detect_signs() is allowed to return.
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_signs(image):
    """Return one label and filled face area for each supported sign."""
    hsv, red, blue, white = _colour_masks(image)
    detections = []
    red_signs, _ = _red_faces(red, white)
    detections.extend({"label": label, "area": area}
                      for label, area, face in red_signs)

    for contour in _contours(blue):
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if area < 40 or min(width, height) < 6:
            continue
        hull = cv2.convexHull(contour)
        if area / max(1.0, cv2.contourArea(hull)) < 0.75:
            continue
        face = _face_mask(contour, blue.shape)
        fraction = cv2.countNonZero(white & face) / max(1, cv2.countNonZero(face))
        if not 0.08 <= fraction <= 0.55:
            continue
        polygon = cv2.approxPolyDP(hull, 0.035 * cv2.arcLength(hull, True), True)
        if len(polygon) != 4:
            continue
        label = _blue_letter(image, polygon)
        if label is not None:
            detections.append({"label": label, "area": int(round(area))})
    return detections


def detect_obstacle(image):
    """Detect the visible red obstacle, including dark and separated faces."""
    # Keep these thresholds separate from the already working sign detector.
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image of shape (height, width, 3)")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_hue = (hue <= 16) | (hue >= 164)
    weak = (red_hue & (saturation >= 45) & (value >= 5)).astype(np.uint8) * 255
    strong = red_hue & (saturation >= 60) & (value >= 10)
    # Hysteresis retains dark faces connected to a confidently red region.
    number, labels = cv2.connectedComponents(weak, connectivity=8)
    seed_labels = labels[strong]
    seed_counts = np.bincount(seed_labels, minlength=number)
    seed_values = np.bincount(seed_labels, weights=value[strong], minlength=number)
    # Bright objects need a higher floor to reject dark JPEG ringing; dark
    # objects retain their shadowed faces. Calibrate each connected region.
    floors = np.ceil(np.maximum(5, 0.08 * seed_values / np.maximum(1, seed_counts))).astype(np.uint8)
    seeded = (seed_counts >= 4).astype(np.uint8) * 255
    seeded[0] = 0
    red = cv2.bitwise_and(seeded[labels], cv2.compare(value, floors[labels], cv2.CMP_GE))
    red = _remove_overlay_dots(red, hsv)
    measured = red.copy()
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    pale = cv2.inRange(hsv, (0, 0, 8), (179, 60, 255))
    red = _obstacle_remove_signs(red, pale, hsv)
    measured &= red
    candidates = []
    for contour in _contours(red):
        area = cv2.contourArea(contour)
        # Below this support, a few coloured texture pixels are ambiguous.
        # This cutoff is absolute; resolution no longer raises it.
        if area < 40:
            continue
        rectangle = cv2.minAreaRect(contour)
        sides = rectangle[1]
        short, long = sorted(sides)
        aspect = short / max(1.0, long)
        if short < 4 or aspect < 0.12:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        # Use the object's rotated box, so camera roll and sloping faces
        # do not make a genuine obstacle fail an axis-aligned fill ratio.
        box_fill = area / max(1.0, short * long)
        if box_fill < 0.4:
            continue
        solidity = area / max(1.0, hull_area)
        if solidity < 0.5 or _cross_like(contour, short):
            continue
        polygon = cv2.approxPolyDP(hull, 0.018 * cv2.arcLength(hull, True), True)
        # A block has long straight edges. Round drawing markers and the
        # concave hospital cross must not become obstacle detections.
        coarse = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
        coarse_fit = cv2.contourArea(coarse) / max(1.0, hull_area)
        quadrilateral = len(coarse) == 4 and coarse_fit >= 0.85
        x, y, width, height = cv2.boundingRect(contour)
        at_border = (x == 0 or y == 0 or x + width == image.shape[1] or
                     y + height == image.shape[0])
        clipped_face = at_border and 3 <= len(coarse) <= 5 and coarse_fit >= 0.75
        if short < 7 and not (quadrilateral and aspect >= 0.55):
            continue
        if not (quadrilateral or clipped_face) and not (box_fill >= 0.8 and 4 <= len(polygon) <= 6):
            continue
        box, face = _contour_patch(contour)
        x, y, width, height = box
        count = cv2.countNonZero(face)
        if cv2.countNonZero(face & pale[y:y + height, x:x + width]) / max(1, count) > 0.5:
            continue
        if cv2.countNonZero(face & red[y:y + height, x:x + width]) / max(1, count) < 0.5:
            continue
        # Closing repairs connectivity only. Count original supported pixels
        # so a repaired gap or an occluding pole cannot inflate the area.
        anchor = aspect >= 0.3 or (clipped_face and solidity >= 0.6)
        candidates.append((box, cv2.countNonZero(face & measured[y:y + height, x:x + width]),
                           anchor))
    area = _obstacle_group_area(candidates)
    return {"present": bool(area), "area": int(area)}


def _contour_patch(contour):
    """Rasterize only a contour's bounding rectangle, not the whole image."""
    x, y, width, height = cv2.boundingRect(contour)
    patch = np.zeros((height, width), np.uint8)
    shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
    cv2.drawContours(patch, [shifted], -1, 255, cv2.FILLED)
    return (x, y, width, height), patch


def _cross_like(contour, short_side):
    """A hospital cross has four deep notches; an occluded block has fewer."""
    if len(contour) < 4:
        return False
    indices = cv2.convexHull(contour, returnPoints=False)
    if len(indices) < 3:
        return False
    defects = cv2.convexityDefects(contour, indices)
    if defects is None:
        return False
    depths = defects[:, 0, 3] / 256.0
    return np.count_nonzero(depths > max(2, 0.14 * short_side)) >= 3


def _remove_overlay_dots(red, hsv):
    """Remove solid bright round annotation dots even when touching a body."""
    bright = cv2.inRange(hsv, (0, 210, 248), (179, 255, 255)) & red
    for contour in _contours(bright):
        area = cv2.contourArea(contour)
        if area < 6:
            continue
        short, long = sorted(cv2.minAreaRect(contour)[1])
        if short < 3 or short / max(1.0, long) < 0.65:
            continue
        fill = area / max(1, short * long)
        hull = cv2.convexHull(contour)
        polygon = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True)
        if not 0.60 <= fill <= 0.87 or len(polygon) < 5:
            continue
        (x, y, width, height), face = _contour_patch(contour)
        if cv2.countNonZero(face & bright[y:y + height, x:x + width]) / cv2.countNonZero(face) < 0.9:
            continue
        # The bright core excludes anti-aliased boundary pixels. Extend its
        # silhouette by one pixel to remove that fringe as well.
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(red.shape[1], x + width + 1), min(red.shape[0], y + height + 1)
        patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(patch, [contour - np.array([[[x0, y0]]], dtype=contour.dtype)],
                         -1, 255, cv2.FILLED)
        patch = cv2.dilate(patch, np.ones((3, 3), np.uint8))
        red[y0:y1, x0:x1][patch > 0] = 0
    return red


def _obstacle_remove_signs(red, pale, hsv):
    """Exclude sign rims without treating pale occlusions as road signs."""
    contours, hierarchy = cv2.findContours(red, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return red
    parents = {int(hierarchy[0, index, 3]) for index, contour in enumerate(contours)
               if hierarchy[0, index, 3] >= 0 and cv2.contourArea(contour) >= 20}
    for parent in parents:
        x, y, width, height = cv2.boundingRect(contours[parent])
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(red.shape[1], x + width + 1), min(red.shape[0], y + height + 1)
        region = red[y0:y1, x0:x1]
        neutral = pale[y0:y1, x0:x1]
        signs, _ = _red_faces(region, neutral)
        value = hsv[y0:y1, x0:x1, 2]
        for label, area, face in signs:
            neutral_values = value[(neutral > 0) & (face > 0)]
            if not neutral_values.size:
                continue
            # Real red signs contain a dark pictogram/digits on their pale
            # interior. A plain grey/white occluder inside a red obstacle
            # does not supply that evidence.
            dark = (face > 0) & (value < 0.6 * np.median(neutral_values))
            if np.count_nonzero(dark) < max(2, 0.015 * cv2.countNonZero(face)):
                continue
            region[face > 0] = 0
    return red


def _obstacle_group_area(candidates):
    """Sum close, aligned fragments; never fill the occluding gap."""
    parents = list(range(len(candidates)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, (first, _, _) in enumerate(candidates):
        x, y, width, height = first
        for j in range(i):
            xx, yy, other_width, other_height = candidates[j][0]
            horizontal_gap = max(0, x - xx - other_width, xx - x - width)
            vertical_gap = max(0, y - yy - other_height, yy - y - height)
            horizontal_overlap = max(0, min(x + width, xx + other_width) - max(x, xx))
            vertical_overlap = max(0, min(y + height, yy + other_height) - max(y, yy))
            limit = max(2, 0.30 * max(width, height, other_width, other_height))
            aligned = ((vertical_overlap >= 0.6 * min(height, other_height) and
                        horizontal_gap <= limit) or
                       (horizontal_overlap >= 0.6 * min(width, other_width) and
                        vertical_gap <= limit))
            if aligned:
                parents[root(i)] = root(j)
    totals = {}
    anchors = {}
    boxes = {}
    counts = {}
    for i, (box, area, anchor) in enumerate(candidates):
        group = root(i)
        totals[group] = totals.get(group, 0) + area
        anchors[group] = anchors.get(group, False) or anchor
        counts[group] = counts.get(group, 0) + 1
        x, y, width, height = box
        old = boxes.get(group, (x, y, x + width, y + height))
        boxes[group] = (min(old[0], x), min(old[1], y),
                        max(old[2], x + width), max(old[3], y + height))
    valid = []
    for group, area in totals.items():
        x, y, right, bottom = boxes[group]
        width, height = right - x, bottom - y
        # Thin disconnected faces are accepted as part of a body, but an
        # isolated red line is insufficient evidence for an obstacle.
        assembled = (counts[group] > 1 and
                     min(width, height) / max(width, height) >= 0.3 and
                     area / (width * height) >= 0.6)
        if anchors[group] or assembled:
            valid.append(area)
    return max(valid, default=0)


# Helpers: _colour_masks, _contours, _face_mask, _blue_letter, _red_faces,
#          _obstacle_group_area, _contour_patch, _cross_like,
#          _remove_overlay_dots, _obstacle_remove_signs.
# Additional global variables: None. All helpers are inside this block.
def _colour_masks(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image of shape (height, width, 3)")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 90, 45), (12, 255, 255)) |
           cv2.inRange(hsv, (168, 90, 45), (179, 255, 255)))
    blue = cv2.inRange(hsv, (90, 80, 40), (135, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 110), (179, 85, 255))
    # Repair red rims without moving them. Keep the blue mask unchanged:
    # closing can join the pale border on a small, partly clipped sign.
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return hsv, red, blue, white


def _contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def _face_mask(contour, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask


def _red_faces(red, white):
    """Identify hollow red signs and separate them from connected solids."""
    contours, hierarchy = cv2.findContours(red, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
    remaining = red.copy()
    signs = []
    if hierarchy is None:
        return signs, remaining
    for index, inner in enumerate(contours):
        parent_index = hierarchy[0, index, 3]
        if parent_index < 0 or cv2.contourArea(inner) < 20:
            continue
        inside = _face_mask(inner, red.shape)
        if cv2.countNonZero(inside & white) / max(1, cv2.countNonZero(inside)) < 0.45:
            continue
        hull = cv2.convexHull(inner)
        if cv2.contourArea(inner) / max(1.0, cv2.contourArea(hull)) < 0.85:
            continue
        polygon = cv2.approxPolyDP(hull, 0.035 * cv2.arcLength(hull, True), True)
        if len(polygon) == 3:
            label = "School"
        elif len(polygon) >= 5:
            label = "Speed40"
        else:
            continue
        outer = contours[parent_index]
        # An isolated rim has the same shape as its interior. If an
        # obstacle joins it, reconstruct the face using the rim thickness
        # observed along unobstructed rays, rather than the merged outline.
        if cv2.matchShapes(outer, inner, cv2.CONTOURS_MATCH_I1, 0) > 0.055:
            moments = cv2.moments(inner)
            center = np.array([moments['m10'], moments['m01']]) / moments['m00']
            points = hull.reshape(-1, 2).astype(float)
            factors = []
            for point in points:
                offset = point - center
                radius = np.linalg.norm(offset)
                if radius < 2:
                    continue
                # Scan outward with subpixel steps. Rays blocked by the
                # obstacle get the maximum factor; the median ignores them.
                factor = 1.5
                for scale in np.arange(1.02, 1.5, 0.01):
                    px, py = np.rint(center + offset * scale).astype(int)
                    if not (0 <= py < red.shape[0] and 0 <= px < red.shape[1]) or red[py, px] == 0:
                        factor = scale
                        break
                factors.append(factor)
            if not factors:
                continue
            expansion = float(np.median(factors))
            outer = np.rint(center + (points - center) * expansion).astype(np.int32)
            outer = outer.reshape(-1, 1, 2)
        face = _face_mask(outer, red.shape)
        signs.append((label, int(round(cv2.contourArea(outer))), face))
        remaining[face > 0] = 0
    return signs, remaining


def _blue_letter(image, polygon):
    # Rectify the sign's quadrilateral before comparing letter geometry.
    points = polygon.reshape(4, 2).astype(np.float32)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1],
                                       points[:, 0] - center[0]))]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    target = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], np.float32)
    transform = cv2.getPerspectiveTransform(points, target)
    sign = cv2.warpPerspective(image, transform, (80, 80))
    hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)
    pale = cv2.inRange(hsv, (0, 0, 110), (179, 85, 255))
    pale[:8, :] = pale[-8:, :] = 0
    pale[:, :8] = pale[:, -8:] = 0
    candidates = _contours(pale)
    if not candidates:
        return None
    glyph = max(candidates, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(glyph)
    if height < 22 or width < 10 or cv2.contourArea(glyph) < 80:
        return None
    # Only retain the selected component; preserve its holes (the P bowl).
    component = _face_mask(glyph, pale.shape) & pale
    letter = cv2.resize(component[y:y + height, x:x + width], (40, 56),
                        interpolation=cv2.INTER_NEAREST) > 0
    left_stem = float(letter[:, :10].mean())
    lower_right = float(letter[34:, 27:].mean())
    upper_right = float(letter[:22, 27:].mean())
    if left_stem < 0.4 or upper_right < 0.25:
        return None
    if lower_right > 0.45:
        return "Hospital"
    if lower_right < 0.22:
        return "Parking"
    return None


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_signs(detections, image_name):
    '''
    Purpose:
    ---
    Check that detect_signs() returned the expected structure and normalise
    it, so a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `detections` :  [ object ]     whatever detect_signs() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ list of dict ]   [{"label": str, "area": int}, ...]
    '''
    where = "detect_signs() on {}".format(image_name)

    if not isinstance(detections, list):
        raise TypeError("{} must return a list, got {}".format(where, type(detections).__name__))

    clean = []
    for index, entry in enumerate(detections):
        entry_where = "{}, entry {}".format(where, index)

        if not isinstance(entry, dict):
            raise TypeError("{} must be a dict, got {}".format(entry_where, type(entry).__name__))

        missing = {"label", "area"} - set(entry.keys())
        if missing:
            raise ValueError("{} is missing the key(s): {}".format(entry_where, ", ".join(sorted(missing))))

        label = entry["label"]
        if label not in SIGN_LABELS:
            raise ValueError("{} returned label = '{}', expected one of {}".format(
                entry_where, label, ", ".join(SIGN_LABELS)))

        area = _validate_area(entry["area"], entry_where)
        clean.append({"label": label, "area": area})

    return clean


def validate_obstacle(result, image_name):
    '''
    Purpose:
    ---
    Check that detect_obstacle() returned the expected structure and
    normalise it.

    Input Arguments:
    ---
    `result` :      [ object ]     whatever detect_obstacle() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ dict ]       {"present": bool, "area": int}
    '''
    where = "detect_obstacle() on {}".format(image_name)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"present", "area"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    present = result["present"]
    if not isinstance(present, (bool, np.bool_)):
        raise TypeError("{} returned present of type {}, expected a bool".format(
            where, type(present).__name__))
    present = bool(present)

    area = _validate_area(result["area"], where)
    if not present and area != 0:
        raise ValueError("{} returned present=False but area={}, expected 0".format(where, area))

    return {"present": present, "area": area}


def _validate_area(area, where):
    if isinstance(area, bool) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned area of type {}, expected a number".format(
            where, type(area).__name__))
    area = int(round(float(area)))
    if area < 0:
        raise ValueError("{} returned a negative area: {}".format(where, area))
    return area
