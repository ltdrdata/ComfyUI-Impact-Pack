export function getFileItem(baseType, value) {
    if (typeof value !== "string" || !value) return null;
    const annotation = /\s*\[(input|output|temp)\]\s*$/i.exec(value);
    const path = (annotation ? value.slice(0, annotation.index) : value).replaceAll("\\", "/");
    const slash = path.lastIndexOf("/");
    return {
        filename: path.slice(slash + 1),
        subfolder: path.slice(0, Math.max(slash, 0)),
        type: annotation ? annotation[1].toLowerCase() : baseType
    };
}

// The mask editor and the SAM Detector open the picture a node shows. A node gets its picture when the node itself is
// executed, which a run of the workflow does not guarantee: an unconnected input, a bypass or a switch keeps a run from
// reaching it. A node that was not executed, or whose file is gone, shows none.
export function showsImage(node) {
    const image = node.imgs?.[node.imageIndex ?? 0];
    return !!image?.src && !(image.complete && !image.naturalWidth);
}
