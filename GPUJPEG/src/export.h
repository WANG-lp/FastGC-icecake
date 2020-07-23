#pragma once

/** JPEG table for huffman decoding */
struct gpujpeg_table_huffman_decoder {
    /// Smallest code of length k
    int mincode[17];
    /// Largest code of length k (-1 if none)
    int maxcode[18];
    /// Huffval[] index of 1st symbol of length k
    int valptr[17];
    /// # bits, or 0 if too long
    int look_nbits[256];
    /// Symbol, or unused
    unsigned char look_sym[256];

    /// These two fields directly represent the contents of a JPEG DHT marker
    /// bits[k] = # of symbols with codes of
    unsigned char bits[17];
    /// The symbols, in order of incr code length
    unsigned char huffval[256];
};

/**
 * JPEG segment structure. Segment is data in scan generated by huffman coder
 * for N consecutive MCUs, where N is restart interval (e.g. data for MCUs between
 * restart markers)
 */
struct gpujpeg_segment {
    /// Scan index (in which segment belongs)
    int scan_index;
    /// Segment index in the scan (position of segment in scan starting at 0)
    int scan_segment_index;
    /// MCU count in segment
    int mcu_count;

    /// Data compressed index (output/input data from/to segment for encoder/decoder)
    int data_compressed_index;
    /// Date temp index (temporary data of segment in CC 2.0 encoder)
    int data_temp_index;
    /// Data compressed size (output/input data from/to segment for encoder/decoder)
    int data_compressed_size;

    /// Offset of first block index
    int block_index_list_begin;
    /// Number of blocks of the segment
    int block_count;
};