"""NuGraph3 data transform"""
import torch
from torch_geometric.transforms import BaseTransform
from pynuml.data import NuGraphData

class Transform(BaseTransform):
    """
    NuGraph3 data transform
    
    Args:
        planes: Tuple of detector plane names
    """
    def __init__(self,
                 planes: tuple[str],
                 use_pmt_pmt_edges: bool = True,
                 use_legacy_sp_pmt_edges: bool = False,
                 use_ophit_ophit_edges: bool = True,
                 ophit_pmt_neighbor_radius: float | None = None,
                 ophit_pmt_neighbor_radius_scale: float = 1.2,
                 pmt_sp_radius_scale: float = 1.2):
        super().__init__()
        self.planes = planes
        self.use_pmt_pmt_edges = use_pmt_pmt_edges
        self.use_legacy_sp_pmt_edges = use_legacy_sp_pmt_edges
        self.use_ophit_ophit_edges = use_ophit_ophit_edges
        self.ophit_pmt_neighbor_radius = ophit_pmt_neighbor_radius
        self.ophit_pmt_neighbor_radius_scale = ophit_pmt_neighbor_radius_scale
        self.pmt_sp_radius_scale = pmt_sp_radius_scale

    def forward(self, data: NuGraphData) -> NuGraphData:

        """
        Apply transform for compatibility with NuGraph3 model

        Args:
           data: NuGraph data object to transform
        """

        # transform old planar format into new hierarchical format
        if "hit" not in data.node_types:

            # unify planar edges
            edge_plane = []
            edge_nexus = []
            for i, p in enumerate(self.planes):
                offset = 0
                for j in range(i): # get offset from previous planes
                    offset += data[self.planes[j]].num_nodes
                edge_plane.append(data[p, "plane", p].edge_index + offset)
                del data[p, "plane", p]
                edge_nexus.append(data[p, "nexus", "sp"].edge_index)
                edge_nexus[-1][0] += offset # increment only the plane node index
                del data[p, "nexus", "sp"]
            data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edge_plane, dim=1)
            data["hit", "nexus", "sp"].edge_index = torch.cat(edge_nexus, dim=1)

            # add plane index to feature tensor
            for i, p in enumerate(self.planes):
                data[p].plane = torch.empty_like(data[p].x[:,0], dtype=int).fill_(i)
                data[p].x = torch.cat([data[p].x, data[p].plane.unsqueeze(1)], dim=1)

            # merge planar node stores
            for attr in data[self.planes[0]].node_attrs():
                data["hit"][attr] = torch.cat([data[p][attr] for p in self.planes], dim=0)
            for p in self.planes:
                del data[p]

            # add true instance nodes
            if hasattr(data["hit"], "y_instance"):
                y = data["hit"].y_instance
                mask = y != -1
                y = y[mask]
                instances = y.unique()
                # remap instances
                imax = instances.max() + 1 if instances.size(0) else 0
                if instances.size(0) != imax:
                    remap = torch.full((imax,), -1, dtype=torch.long)
                    remap[instances] = torch.arange(instances.size(0))
                    y = remap[y]
                data["particle-truth"].x = torch.empty(instances.size(0), 0)
                edges = torch.stack((mask.nonzero().squeeze(1), y), dim=0).long()
                data["hit", "cluster-truth", "particle-truth"].edge_index = edges
                del data["hit"].y_instance

            # add edges to and from event node
            data["evt"].x = torch.empty((1, 0))
            lo = torch.arange(data["hit"].num_nodes, dtype=torch.long)
            hi = torch.zeros(data["hit"].num_nodes, dtype=torch.long)
            data["hit", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)
            lo = torch.arange(data["sp"].num_nodes, dtype=torch.long)
            hi = torch.zeros(data["sp"].num_nodes, dtype=torch.long)
            data["sp", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)

        # rename true hit position (remove once spacepoint decoder is mature)
        if "c" in data["hit"].keys():
            data["hit"].y_position = data["hit"].c
            del data["hit"].c

        # ensure event truth labels have correct format
        evt = data["evt"]
        if not evt.y.ndim:
            evt.y = evt.y.reshape([1])

        # concatenate position tensor onto node features
        h = data["hit"]
        h.x = torch.cat((h.pos, h.x), dim=-1)

        for edge_type in [("flash", "in", "evt"), ("ophit", "in", "pmt")]:
            if edge_type in data.edge_types:
                if data[edge_type].edge_index.dim() == 1:
                    data[edge_type].edge_index = data[edge_type].edge_index.unsqueeze(1)

        # build pmt-pmt and pmt-sp edges
        if "pmt" in data.node_types:
            pmt_pos = getattr(data["pmt"], "pos", None)
            n_pmt = data["pmt"].num_nodes

            # build pmt-pmt edges with KNN (k=5)
            if self.use_pmt_pmt_edges and pmt_pos is not None and n_pmt > 1:
                distances = torch.cdist(pmt_pos, pmt_pos, p=2)
                distances.fill_diagonal_(float("inf"))
                knn = min(5, n_pmt - 1)
                _, neighbor_idx = torch.topk(distances, knn, largest=False, dim=1)
                source = torch.arange(n_pmt, device=neighbor_idx.device, dtype=torch.long).repeat_interleave(knn)
                target = neighbor_idx.reshape(-1)
                pmt_edge = torch.stack((source, target), dim=0)
                pmt_edge = torch.cat((pmt_edge, pmt_edge.flip(0)), dim=1)
            else:
                device = pmt_pos.device if pmt_pos is not None else None
                pmt_edge = torch.empty((2, 0), dtype=torch.long, device=device)
            data["pmt", "knn", "pmt"].edge_index = pmt_edge.long()

            # build pmt-sp edges: per-PMT radius = nearest-SP distance * pmt_sp_radius_scale
            sp_pos = data["sp"].pos if "sp" in data.node_types and hasattr(data["sp"], "pos") else None
            if not self.use_legacy_sp_pmt_edges and pmt_pos is not None and sp_pos is not None and n_pmt > 0 and data["sp"].num_nodes > 0:
                common_dim = min(pmt_pos.size(-1), sp_pos.size(-1))
                pmt_metric = pmt_pos[:, -common_dim:]
                sp_metric = sp_pos[:, -common_dim:]
                distances = torch.cdist(pmt_metric, sp_metric, p=2)

                min_dist = distances.min(dim=1).values  # each PMT's nearest-SP distance
                radii = min_dist * self.pmt_sp_radius_scale  # [n_pmt], per-PMT threshold
                data["pmt"].pmt_sp_radius = radii.detach()

                edge_mask = distances <= radii.unsqueeze(1)
                pmt_indices, sp_indices = edge_mask.nonzero(as_tuple=True)
                pmt_sp_edges = torch.stack((pmt_indices, sp_indices), dim=0)
                pmt_sp_distances = distances[pmt_indices, sp_indices]

                sp_degree = torch.bincount(sp_indices.long(), minlength=data["sp"].num_nodes)
            else:
                device = pmt_pos.device if pmt_pos is not None else None
                pmt_sp_edges = torch.empty((2, 0), dtype=torch.long, device=device)
                pmt_sp_distances = torch.empty((0,), dtype=torch.float, device=device)
                if "sp" in data.node_types:
                    sp_degree = torch.zeros(data["sp"].num_nodes, dtype=torch.long, device=device)
            data["pmt", "knn", "sp"].edge_index = pmt_sp_edges.long()
            data["pmt", "knn", "sp"].edge_distance = pmt_sp_distances.float()
            if "sp" in data.node_types:
                data["sp"].pmt_degree = sp_degree.long()

        # build ophit-ophit edges using radius = adjacent_pmt_distance * ophit_pmt_neighbor_radius_scale
        if self.use_ophit_ophit_edges and "ophit" in data.node_types and ("ophit", "in", "pmt") in data.edge_types:
            ophit_in_pmt = data["ophit", "in", "pmt"].edge_index
            if ophit_in_pmt.dim() == 1:
                ophit_in_pmt = ophit_in_pmt.unsqueeze(1)

            if ophit_in_pmt.numel() == 0:
                ophit_edges = torch.empty((2, 0), dtype=torch.long, device=ophit_in_pmt.device)
            else:
                ophit_idx = ophit_in_pmt[0].long()
                pmt_idx = ophit_in_pmt[1].long()

                n_pmt = data["pmt"].num_nodes if "pmt" in data.node_types else 0
                pmt_pos = getattr(data["pmt"], "pos", None) if "pmt" in data.node_types else None
                if n_pmt > 0 and pmt_pos is not None:
                    pmt_pos_metric = pmt_pos.float()
                    pmt_distances = torch.cdist(pmt_pos_metric, pmt_pos_metric, p=2)

                    if self.ophit_pmt_neighbor_radius is None:
                        if n_pmt > 1:
                            pmt_distances_for_nn = pmt_distances.clone()
                            pmt_distances_for_nn.fill_diagonal_(float("inf"))
                            nearest = pmt_distances_for_nn.min(dim=1).values
                            # per-PMT radius: each PMT's nearest-neighbor distance * scale
                            radii = nearest * self.ophit_pmt_neighbor_radius_scale
                        else:
                            radii = torch.zeros(n_pmt, dtype=pmt_distances.dtype, device=pmt_distances.device)
                        pmt_neighbors = pmt_distances <= radii.unsqueeze(1)
                        data["pmt"].ophit_neighbor_radius = radii.detach()
                    else:
                        radius = torch.tensor(self.ophit_pmt_neighbor_radius,
                                              dtype=pmt_distances.dtype,
                                              device=pmt_distances.device)
                        pmt_neighbors = pmt_distances <= radius
                        data["pmt"].ophit_neighbor_radius = radius.detach().reshape(1)
                    pmt_neighbors.fill_diagonal_(True)

                    data["pmt"].ophit_neighbor_pmt_count = (
                        pmt_neighbors.sum(dim=1) - 1).long()

                    pmt_to_ophits = [torch.empty((0,), dtype=torch.long, device=ophit_idx.device)
                                     for _ in range(n_pmt)]
                    for pmt in pmt_idx.unique(sorted=True):
                        members = ophit_idx[pmt_idx == pmt].unique(sorted=True)
                        pmt_to_ophits[int(pmt.item())] = members

                    edge_blocks = []
                    for src_ophit, src_pmt in zip(ophit_idx, pmt_idx):
                        neighbor_pmts = torch.nonzero(pmt_neighbors[src_pmt], as_tuple=False).squeeze(1)
                        neighbor_ophits = [pmt_to_ophits[int(p.item())] for p in neighbor_pmts]
                        neighbor_ophits = [hits for hits in neighbor_ophits if hits.numel() > 0]
                        if not neighbor_ophits:
                            continue
                        dst = torch.cat(neighbor_ophits, dim=0)
                        src = src_ophit.repeat(dst.numel())
                        mask = src != dst
                        if mask.any():
                            edge_blocks.append(torch.stack((src[mask], dst[mask]), dim=0))

                    if edge_blocks:
                        ophit_edges = torch.cat(edge_blocks, dim=1)
                    else:
                        ophit_edges = torch.empty((2, 0), dtype=torch.long, device=ophit_in_pmt.device)
                else:
                    # when pmt geometry is missing, build edges within same pmt only
                    edge_blocks = []
                    for pmt in pmt_idx.unique(sorted=True):
                        members = ophit_idx[pmt_idx == pmt].unique(sorted=True)
                        count = members.numel()
                        if count <= 1:
                            continue
                        src = members.repeat_interleave(count)
                        dst = members.repeat(count)
                        mask = src != dst
                        edge_blocks.append(torch.stack((src[mask], dst[mask]), dim=0))

                    if edge_blocks:
                        ophit_edges = torch.cat(edge_blocks, dim=1)
                    else:
                        ophit_edges = torch.empty((2, 0), dtype=torch.long, device=ophit_in_pmt.device)

            data["ophit", "knn", "ophit"].edge_index = ophit_edges.long()
        elif "ophit" in data.node_types:
            data["ophit", "knn", "ophit"].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data