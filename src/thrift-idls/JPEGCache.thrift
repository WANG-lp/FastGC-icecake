namespace cpp JPEGCache


service JPEGCache {
    binary get(1:string filename)
    binary getWithROI(1:string filename, 2:i32 offset_x, 3:i32 offset_y, 4:i32 roi_w, 5:i32 roi_h)
    i32 put(1:string filename, 2:binary content)
}
