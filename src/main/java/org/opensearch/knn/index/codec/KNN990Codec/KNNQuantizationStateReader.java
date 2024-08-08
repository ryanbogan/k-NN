/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.query.QuantizationStateCache;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KNNQuantizationStateReader
{
    private static class FieldQuantizationState {
        String fieldName;
        byte[] stateBytes;

        FieldQuantizationState(String fieldName, byte[] stateBytes) {
            this.fieldName = fieldName;
            this.stateBytes = stateBytes;
        }
    }


    public static void read(SegmentReadState state)  {
        String quantizationStateFileName =
                IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "qs");


        try (IndexInput input = state.directory.openInput(quantizationStateFileName, IOContext.READ)) {

            long footerStart = input.length() - CodecUtil.footerLength();
            long markerAndIndexPosition = footerStart - Integer.BYTES - Long.BYTES;
            input.seek(markerAndIndexPosition);
            long indexStartPosition = input.readLong();
            int marker = input.readInt();
            input.seek(indexStartPosition);
            int numFields = input.readInt();

            List<String> fieldNames = new ArrayList<>();
            List<FieldQuantizationState> fieldQuantizationStates = new ArrayList<>();
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
                fieldQuantizationStates.add(new FieldQuantizationState(fieldNames.get(i), stateBytes));
            }
            for (FieldQuantizationState fieldQuantizationState : fieldQuantizationStates) {
                // Deserialize the byte array to a quantization state object
                OneBitScalarQuantizationState quantizationState = OneBitScalarQuantizationState.fromByteArray(fieldQuantizationState.stateBytes);
                QuantizationStateCache.getInstance().addQuantizationState(fieldQuantizationState.fieldName, quantizationState);
            }
        } catch (ClassNotFoundException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}
