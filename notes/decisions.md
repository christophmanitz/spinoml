## 2026-06-18 — protein_seq-Branch: Manifest-Collate verlangt PyG-Data
SpinoMLs graph_collate ruft fuer JEDE Manifest-Branch torch_geometric Batch.from_data_list auf.
Reine Tensor-.pt-Dateien (erste Tokenizer-Version) brechen mit
'TensorBatch' object has no attribute 'stores_as'.
Loesung: protein_seq als PyG-Data-Kettengraph speichern (x=token-ids [L,1], chain edge_index),
ProteinSeqEncoder (Custom node llm2) nimmt den Batch entgegen und nutzt to_dense_batch ->
[B,Lmax] -> Embedding+Conv1d(DeepDTA)+masked GlobalMaxPool. Kein Padding mehr im Tokenizer.
Tokenizer MUSS mit .spinoml/venv/bin/python laufen (torch_geometric). Merke: plain-tensor-Branches
sind ueber das Manifest NICHT moeglich — alles muss ein Graph/Data sein.
