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
