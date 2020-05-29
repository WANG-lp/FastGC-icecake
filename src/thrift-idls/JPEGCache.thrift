namespace cpp JPEGCache


service JPEGCache {
    i32    set_parameters(1:i32 seed, 2:double s1, 3:double s2, 4:double r1, 5:double r2)
    binary get(1:string filename)
    binary getWithROI(1:string filename, 2:i32 offset_x, 3:i32 offset_y, 4:i32 roi_w, 5:i32 roi_h)
    binary getWithRandomCrop(1:string filename)
    binary getRAW(1:string filename)
    i32 put(1:string filename, 2:binary content)
}
