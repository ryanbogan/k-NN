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
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.*;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NativeEnginesKNNVectorsReader extends KnnVectorsReader {

    private final FlatVectorsReader flatVectorsReader;

    public NativeEnginesKNNVectorsReader(SegmentReadState state, final FlatVectorsReader flatVectorsReader) {
        this.flatVectorsReader = flatVectorsReader;
    }

    private  class FieldQuantizationState {
        String fieldName;
        byte[] stateBytes;

        FieldQuantizationState(String fieldName, byte[] stateBytes) {
            this.fieldName = fieldName;
            this.stateBytes = stateBytes;
        }
    }


    private  OneBitScalarQuantizationState readQuantizationState(SegmentReadState state, String fieldName)  {
        String quantizationFileName =
                IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "qs");


        try (IndexInput input = state.directory.openInput(quantizationFileName, IOContext.READ)) {

            long footerStart = input.length() - CodecUtil.footerLength();
            long markerAndIndexPosition = footerStart - Integer.BYTES - Long.BYTES;
            input.seek(markerAndIndexPosition);
            long indexStartPosition = input.readLong();
            int marker = input.readInt();
            input.seek(indexStartPosition);
            int numFields = input.readInt();
            List<FieldQuantizationState> fieldQuantizationStates = new ArrayList<>();
            List<Long> positions = new ArrayList<>();
            List<Integer> lengths = new ArrayList<>();

            // Read each field's metadata from the index section
            for (int i = 0; i < numFields; i++) {
                String fieldName1 = input.readString();
                int length = input.readInt();
                lengths.add(length);
                long position = input.readVLong();
                positions.add(position);
            }
            for (int i = 0; i < numFields; i++) {
                input.seek(positions.get(i));
                byte[] stateBytes = new byte[lengths.get(i)];
                input.readBytes(stateBytes, 0, lengths.get(i));
                // Deserialize the byte array to a quantization state object
                OneBitScalarQuantizationState quantizationState = OneBitScalarQuantizationState.fromByteArray(stateBytes);
            }
            // Validate footer
            return OneBitScalarQuantizationState.fromByteArray(fieldQuantizationStates.get(0).stateBytes);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Checks consistency of this reader.
     *
     * <p>Note that this may be costly in terms of I/O, e.g. may involve computing a checksum value
     * against large data files.
     *
     */
    @Override
    public void checkIntegrity() throws IOException {
        flatVectorsReader.checkIntegrity();
    }

    /**
     * Returns the {@link FloatVectorValues} for the given {@code field}. The behavior is undefined if
     * the given field doesn't have KNN vectors enabled on its {@link FieldInfo}. The return value is
     * never {@code null}.
     *
     * @param field
     */
    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return flatVectorsReader.getFloatVectorValues(field);
    }

    /**
     * Returns the {@link ByteVectorValues} for the given {@code field}. The behavior is undefined if
     * the given field doesn't have KNN vectors enabled on its {@link FieldInfo}. The return value is
     * never {@code null}.
     *
     * @param field
     */
    @Override
    public ByteVectorValues getByteVectorValues(String field) {
        throw new UnsupportedOperationException("ByteVectorValues is not supported here");
    }

    /**
     * Return the k nearest neighbor documents as determined by comparison of their vector values for
     * this field, to the given vector, by the field's similarity function. The score of each document
     * is derived from the vector similarity in a way that ensures scores are positive and that a
     * larger score corresponds to a higher ranking.
     *
     * <p>The search is allowed to be approximate, meaning the results are not guaranteed to be the
     * true k closest neighbors. For large values of k (for example when k is close to the total
     * number of documents), the search may also retrieve fewer than k documents.
     *
     * <p>The returned {@link TopDocs} will contain a {@link ScoreDoc} for each nearest neighbor, in
     * order of their similarity to the query vector (decreasing scores). The {@link TotalHits}
     * contains the number of documents visited during the search. If the search stopped early because
     * it hit {@code visitedLimit}, it is indicated through the relation {@code
     * TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO}.
     *
     * <p>The behavior is undefined if the given field doesn't have KNN vectors enabled on its {@link
     * FieldInfo}. The return value is never {@code null}.
     *
     * @param field        the vector field to search
     * @param target       the vector-valued query
     * @param knnCollector a KnnResults collector and relevant settings for gathering vector results
     * @param acceptDocs   {@link Bits} that represents the allowed documents to match, or {@code null}
     *                     if they are all allowed to match.
     */
    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        throw new UnsupportedOperationException("Search is not supported with Native Engine Reader");
    }

    /**
     * Return the k nearest neighbor documents as determined by comparison of their vector values for
     * this field, to the given vector, by the field's similarity function. The score of each document
     * is derived from the vector similarity in a way that ensures scores are positive and that a
     * larger score corresponds to a higher ranking.
     *
     * <p>The search is allowed to be approximate, meaning the results are not guaranteed to be the
     * true k closest neighbors. For large values of k (for example when k is close to the total
     * number of documents), the search may also retrieve fewer than k documents.
     *
     * <p>The returned {@link TopDocs} will contain a {@link ScoreDoc} for each nearest neighbor, in
     * order of their similarity to the query vector (decreasing scores). The {@link TotalHits}
     * contains the number of documents visited during the search. If the search stopped early because
     * it hit {@code visitedLimit}, it is indicated through the relation {@code
     * TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO}.
     *
     * <p>The behavior is undefined if the given field doesn't have KNN vectors enabled on its {@link
     * FieldInfo}. The return value is never {@code null}.
     *
     * @param field        the vector field to search
     * @param target       the vector-valued query
     * @param knnCollector a KnnResults collector and relevant settings for gathering vector results
     * @param acceptDocs   {@link Bits} that represents the allowed documents to match, or {@code null}
     *                     if they are all allowed to match.
     */
    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        throw new UnsupportedOperationException("Search is not supported with Native Engine Reader");
    }

    /**
     * Closes this stream and releases any system resources associated
     * with it. If the stream is already closed then invoking this
     * method has no effect.
     *
     * <p> As noted in {@link AutoCloseable#close()}, cases where the
     * close may fail require careful attention. It is strongly advised
     * to relinquish the underlying resources and to internally
     * <em>mark</em> the {@code Closeable} as closed, prior to throwing
     * the {@code IOException}.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsReader);
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
    @Override
    public long ramBytesUsed() {
        return 0;
    }
}
