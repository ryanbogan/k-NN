/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads quantization states
 */
public class KNNQuantizationStateReader {

    /**
     * Read quantization states and return list of fieldNames and bytes
     *
     * File format:
     * Header
     * QS1 state bytes
     * QS2 state bytes
     * Number of quantization states
     * QS1 field name
     * QS1 state bytes length
     * QS1 position of state bytes
     * QS2 field name
     * QS2 state bytes length
     * QS2 position of state bytes
     * Position of index section (where QS1 field name is located)
     * -1 (marker)
     * Footer
     *
     * @param state the read state to read from
     */
    public Map<String, byte[]> read(SegmentReadState state) {
        String quantizationStateFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX
        );
        Map<String, byte[]> readQuantizationStateInfos = new HashMap<>();

        try (IndexInput input = state.directory.openInput(quantizationStateFileName, IOContext.READ)) {

            int numFields = getNumFields(input);

            List<String> fieldNames = new ArrayList<>();
            List<Long> positions = new ArrayList<>();
            List<Integer> lengths = new ArrayList<>();

            // Read each field's metadata from the index section
            for (int i = 0; i < numFields; i++) {
                fieldNames.add(input.readString());
                int length = input.readInt();
                lengths.add(length);
                long position = input.readVLong();
                positions.add(position);
            }
            // Read each field's bytes
            for (int i = 0; i < numFields; i++) {
                input.seek(positions.get(i));
                byte[] stateBytes = new byte[lengths.get(i)];
                input.readBytes(stateBytes, 0, lengths.get(i));
                readQuantizationStateInfos.put(fieldNames.get(i), stateBytes);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return readQuantizationStateInfos;
    }

    /**
     * Reads an individual quantization state for a given field
     * @param directory directory to open input
     * @param segmentName segment name
     * @param segmentSuffix segment suffix
     * @param fieldInfo field information
     * @return quantization state
     */
    public QuantizationState read(Directory directory, String segmentName, String segmentSuffix, FieldInfo fieldInfo) throws IOException {
        String quantizationStateFileName = IndexFileNames.segmentFileName(
            segmentName,
            segmentSuffix,
            KNNConstants.QUANTIZATION_STATE_FILE_SUFFIX
        );
        String fieldName = fieldInfo.getName();

        IndexInput input = directory.openInput(quantizationStateFileName, IOContext.READ);
        CodecUtil.retrieveChecksum(input);
        int numFields = getNumFields(input);

        long position = -1;
        int length = 0;

        // Read each field's metadata from the index section
        for (int i = 0; i < numFields; i++) {
            String tempFieldName = input.readString();
            int tempLength = input.readInt();
            long tempPosition = input.readVLong();
            if (tempFieldName.equals(fieldName)) {
                position = tempPosition;
                length = tempLength;
                break;
            }
        }

        if (position == -1 || length == 0) {
            throw new IllegalArgumentException(String.format("Field %s not found", fieldName));
        }

        input.seek(position);
        byte[] stateBytes = new byte[length];
        input.readBytes(stateBytes, 0, length);
        input.close();
        // Deserialize the byte array to a quantization state object
        // TODO: Get params from field info and deserialize
        return null;
    }

    private int getNumFields(IndexInput input) throws IOException {
        long footerStart = input.length() - CodecUtil.footerLength();
        long markerAndIndexPosition = footerStart - Integer.BYTES - Long.BYTES;
        input.seek(markerAndIndexPosition);
        long indexStartPosition = input.readLong();
        input.readInt();
        input.seek(indexStartPosition);
        return input.readInt();
    }
}
