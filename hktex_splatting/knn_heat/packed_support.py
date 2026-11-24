"""Prepare visible triangle embedding bounds and fixed-size candidate staging.

Kernel candidate selection; geometry is fixed but source support is refreshed.
Main API: PreparedSupport."""
import numpy as np
import torch


class PreparedSupport:
    """Explicit snapshot of fixed mesh/archive/visibility geometry.

    Rebuild when any geometry changes. Sources are supplied afresh to select().
    The runner binds a plan to each immutable camera visibility array.
    """
    @torch.no_grad()
    def __init__(self, faces, embedding, face_ids, neighbors, face_chunk=128, profile=False,
                 *, execution_batch_faces=0):
        ids=np.asarray(face_ids)
        if ids.ndim!=2 or ids.dtype.kind not in 'iu' or np.any(ids < -1) or np.any(ids>=len(faces)):
            raise ValueError('Invalid support visibility')
        if neighbors<1 or face_chunk<1 or embedding.ndim!=2:
            raise ValueError('Invalid support geometry/policy')
        if embedding.requires_grad:
            raise ValueError('Prepared support requires the fixed spectral archive')
        self.faces,self.embedding,self.face_ids=faces,embedding,ids.copy()
        self.versions=(faces._version,embedding._version)
        self.visible=np.unique(ids[ids>=0]).copy()
        self.face_count=len(faces)
        self.device,self.dim=embedding.device,embedding.shape[1]
        self.neighbors,self.face_chunk,self.profile=neighbors,face_chunk,profile
        self.chunks=[]
        for begin in range(0,len(self.visible),face_chunk):
            fi=torch.as_tensor(self.visible[begin:begin+face_chunk],device=self.device)
            corners=embedding[faces[fi].long()]
            center=corners.double().mean(1)
            radius=torch.linalg.vector_norm(corners.double()-center[:,None],dim=-1).amax(1)
            self.chunks.append((begin,center,radius))
        if execution_batch_faces and execution_batch_faces != 512:
            raise ValueError('Audited execution batch must be 512 faces')
        self.execution_chunks = self.chunks
        if execution_batch_faces and self.chunks:
            centers, radii = (
                torch.cat([c for _, c, _ in self.chunks]), torch.cat([r for _, _, r in self.chunks]))
            self.execution_chunks = [(i, centers[i:i+execution_batch_faces], radii[i:i+execution_batch_faces])
                                     for i in range(0, len(self.visible), execution_batch_faces)]
        self.calls=0

    def matches(self,faces,embedding,face_ids,neighbors,face_chunk=128):
        return (faces is self.faces and embedding is self.embedding and
                (faces._version,embedding._version)==self.versions and neighbors==self.neighbors and
                face_chunk==self.face_chunk and np.array_equal(face_ids,self.face_ids))

    @torch.no_grad()
    def candidate_rows(self, sources, pool_cap=128, guard=1e-5):
        """Fresh device rows, shared with the triangle support CSR path.

        Keep the exact float64 operations, chunk size and native topk tie
        policy. Callers own the returned storage; no learned IDs live here.
        """
        n = len(sources)
        width = min(n, pool_cap or 512)
        rows = torch.empty((len(self.visible), width+1), dtype=torch.int32, device=self.device)
        live = sources.double() if self.chunks else None
        source_ids = torch.arange(n, device=self.device) if self.chunks else None
        for begin, center, radius in self.chunks:
            distance = torch.cdist(center, live, compute_mode='donot_use_mm_for_euclid_dist')
            kth = distance.kthvalue(min(self.neighbors, n), dim=1).values
            selected = distance <= kth[:, None]+2*radius[:, None]+guard*(1+kth[:, None]+radius[:, None])
            if pool_cap and pool_cap < n:
                nearest = distance.masked_fill(~selected, float('inf')).topk(pool_cap, dim=1, largest=False).indices
                valid = selected.gather(1, nearest)
                count = valid.sum(1)
                ordered = torch.where(valid, nearest, n).sort(dim=1).values
            else:
                count = selected.sum(1)
                ordered = torch.where(selected, source_ids[None, :], n)
                ordered = (ordered.topk(width, dim=1, largest=False, sorted=True).values
                           if width < n else ordered.sort(dim=1).values)
            rows[begin:begin+len(center), 0] = count
            rows[begin:begin+len(center), 1:] = ordered
        return rows
