/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.List;

public class KNNQuantizationStateWriter {

    static IndexOutput createOutputAndWriteHeader(SegmentWriteState segmentWriteState) throws IOException {
        // Write header and file name
        String quantizationStateFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            "qs"
        );

        IndexOutput output = segmentWriteState.directory.createOutput(quantizationStateFileName, segmentWriteState.context);
        CodecUtil.writeIndexHeader(output, "QuantizationCodec", 0, segmentWriteState.segmentInfo.getId(), segmentWriteState.segmentSuffix);
        return output;
    }

    static FieldQuantizationState writeStateAndReturnFieldQuantizationState(
        IndexOutput output,
        String fieldName,
        QuantizationState quantizationState
    ) throws IOException {
        byte[] stateBytes = quantizationState.toByteArray();
        output.writeBytes(stateBytes, stateBytes.length);
        long position = output.getFilePointer();
        return new FieldQuantizationState(fieldName, stateBytes, position);
    }

    static void writeFooter(IndexOutput output, List<FieldQuantizationState> fieldQuantizationStates) throws IOException {
        // Now write the index section at the end
        long indexStartPosition = output.getFilePointer();
        output.writeInt(fieldQuantizationStates.size());
        for (KNNQuantizationStateWriter.FieldQuantizationState fieldQuantizationState : fieldQuantizationStates) {
            output.writeString(fieldQuantizationState.fieldName);
            output.writeInt(fieldQuantizationState.stateBytes.length);
            output.writeVLong(fieldQuantizationState.position);
        }
        output.writeLong(indexStartPosition);
        output.writeInt(-1);
        CodecUtil.writeFooter(output);
        output.close();
    }

    @AllArgsConstructor
    static class FieldQuantizationState {
        final String fieldName;
        final byte[] stateBytes;
        final Long position;
    }
}
