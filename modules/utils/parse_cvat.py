from lxml import etree
from typing import List


def parse_single_annotation(image):
    """
    Parses single annotation tree and convert to dictionary
    """
    img_dict = {k: v for k, v in image.items() if k in ["id", "name", "width", "height"]}
    # get annotation
    annotations = []
    for anno in image.findall("ellipse"):
        anno_ellipse = {k: v for k, v in anno.items() if k in ["label", "cx", "cy", "rx", "ry"]}
        anno_ellipse["type"] = "ellipse"
        annotations.append(anno_ellipse)
    for anno in image.findall("polyline"):
        anno_polyline = {k: v for k, v in anno.items() if k in ["label", "points"]}
        anno_polyline["points"] = [tuple(l.split(",")) for l in anno_polyline["points"].split(";")]
        anno_polyline["type"] = "polyline"
        annotations.append(anno_polyline)
    img_dict["annotations"] = annotations
    img_dict["image_quality"] = None

    # find tag
    tags = image.findall("tag")
    if tags is not None:
        for tag in tags:
            tag_text = tag.attrib.get("label")
            if tag_text in ["GOOD", "ACCEPTABLE", "POOR"]:
                img_dict["image_quality"] = tag_text
            if tag_text == "GLAUCOMA SUSPECT":
                img_dict["glaucoma_suspect"] = 1
    if "glaucoma_suspect" not in img_dict.keys():
        img_dict["glaucoma_suspect"] = 0
    return img_dict


def parse_cvat_annotation(path: str = "annotations.xml") -> List[dict]:
    """Parses CVAT annotation to a list of dictionaries"""
    root = etree.parse(path).getroot()
    images = root.findall("image")
    annotations = []
    for image in images:
        annotation = parse_single_annotation(image)
        annotations.append(annotation)
    return annotations


def get_filename_to_annotation_dict(annotations: dict, label: str) -> dict:
    filename_to_annotation_dict = {}
    for annotation_dict in annotations:
        for label_dict in annotation_dict["annotations"]:
            if label_dict["label"] == label:
                label_dict["width"] = eval(annotation_dict["width"])
                label_dict["height"] = eval(annotation_dict["height"])
                filename_to_annotation_dict[annotation_dict["name"]] = label_dict
    return filename_to_annotation_dict
