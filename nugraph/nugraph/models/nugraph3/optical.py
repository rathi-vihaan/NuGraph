"""NuGraph optical convolution module"""
import torch
from pynuml.data import NuGraphData
from .core import NuGraphBlock
from .types import TD

class NuGraphOptical(torch.nn.Module):
        """
        NuGraph optical message-passing engine

        This module incorporates optical information into NuGraph

        Args:
        interaction_features: Number of features in interaction embedding
        ophit_features: Number of features in optical hit embedding
        pmt_features: Number of features in PMT (flashsumpe) embedding
        flash_features: Number of features in optical flash embedding
        use_checkpointing: Whether to use checkpointing
        """
        def __init__(self, # pylint: disable=too-many-arguments,too-many-positional-arguments
                     interaction_features: int,
                     nexus_features: int,
                     ophit_features: int,
                     pmt_features: int,
                     flash_features: int,
                     sp_max_degree: int = 12,
                     use_pmt_sp_pruning: bool = True,
                     use_legacy_sp_pmt_edges: bool = False,
                     use_checkpointing: bool = True):
                super().__init__()

                self.use_checkpointing = use_checkpointing
                self.use_pmt_sp_pruning = use_pmt_sp_pruning
                self.use_legacy_sp_pmt_edges = use_legacy_sp_pmt_edges
                self.sp_max_degree = sp_max_degree

                # hierarchical message-passing for optical system
                self.ophit_to_pmt = NuGraphBlock(ophit_features, pmt_features, pmt_features)
                self.pmt_to_flash = NuGraphBlock(pmt_features, flash_features, flash_features)
                self.flash_to_interaction = NuGraphBlock(flash_features,
                                                         interaction_features,
                                                         interaction_features)
                self.interaction_to_flash = NuGraphBlock(interaction_features,
                                                         flash_features, flash_features)
                self.flash_to_pmt = NuGraphBlock(flash_features, pmt_features, pmt_features)
                self.pmt_to_ophit = NuGraphBlock(pmt_features, ophit_features, ophit_features)
                
                # horizontal message-passing
                self.pmt_to_pmt = NuGraphBlock(pmt_features, pmt_features, pmt_features)
                self.ophit_to_ophit = NuGraphBlock(ophit_features, ophit_features, ophit_features)

                # cross-hierarchy message-passing
                self.nexus_to_pmt = NuGraphBlock(nexus_features, pmt_features, pmt_features)
                self.pmt_to_nexus = NuGraphBlock(pmt_features, nexus_features, nexus_features)

        def pmt_sp_pruning_mask(self, data: NuGraphData, edge_index: torch.Tensor,
                                 edge_distance: torch.Tensor) -> torch.Tensor:
                """Returns hard degree-cap mask for pmt-sp edges."""
                if edge_index.numel() == 0:
                        return torch.empty(0, dtype=torch.float, device=edge_index.device)

                dst = edge_index[1].long()
                mask = torch.zeros(edge_distance.size(0), dtype=torch.float, device=edge_distance.device)

                for sp in dst.unique(sorted=True):
                        group_idx = (dst == sp).nonzero(as_tuple=False).squeeze(1)
                        if group_idx.numel() == 0:
                                continue
                        order = torch.argsort(edge_distance[group_idx])
                        keep = order[:self.sp_max_degree]
                        mask[group_idx[keep]] = 1.0

                if "sp" in data.node_types:
                        data["sp"].pmt_degree_cap = torch.tensor(
                                [self.sp_max_degree], dtype=torch.float, device=edge_distance.device)
                return mask

        def checkpoint(self, net: torch.nn.Module, *args) -> TD:
                """
                Checkpoint module, if enabled.

                Args:
                net: Network module
                args: Arguments to network module
                """
                if self.use_checkpointing and self.training:
                        return torch.utils.checkpoint.checkpoint(net, *args, use_reentrant=False)
                return net(*args)

        def forward(self, data: NuGraphData) -> None:
                """
                NuGraphOptical forward pass

                Args:
                data: Graph data object
                """

                # message-passing from ophit to pmt
                data["pmt"].x = self.checkpoint(
                        self.ophit_to_pmt, (data["ophit"].x, data["pmt"].x),
                        data["ophit", "in", "pmt"].edge_index)

                # message-passing from ophit to ophit
                ophit_edges = data["ophit", "knn", "ophit"]
                if "edge_index" in ophit_edges and ophit_edges.edge_index.numel() > 0:
                        data["ophit"].x = self.checkpoint(
                        self.ophit_to_ophit, (data["ophit"].x, data["ophit"].x),
                        ophit_edges.edge_index)

                # message-passing from PMTs to PMTs
                pmt_edges = data["pmt", "knn", "pmt"]
                if "edge_index" in pmt_edges and pmt_edges.edge_index.numel() > 0:
                        data["pmt"].x = self.checkpoint(
                        self.pmt_to_pmt, (data["pmt"].x, data["pmt"].x),
                        pmt_edges.edge_index)

                # message-passing from pmt to spacepoint
                pmt_sp_edges = data["pmt", "knn", "sp"]
                if not self.use_legacy_sp_pmt_edges and "edge_index" in pmt_sp_edges and pmt_sp_edges.edge_index.numel() > 0:
                        edge_distance = getattr(pmt_sp_edges, "edge_distance", None)
                        if edge_distance is None or edge_distance.numel() != pmt_sp_edges.edge_index.size(1):
                                pmt_pos = data["pmt"].pos[pmt_sp_edges.edge_index[0]]
                                sp_pos = data["sp"].pos[pmt_sp_edges.edge_index[1]]
                                common_dim = min(pmt_pos.size(-1), sp_pos.size(-1))
                                edge_distance = torch.linalg.norm(
                                        pmt_pos[:, -common_dim:] - sp_pos[:, -common_dim:], dim=1)

                        if self.use_pmt_sp_pruning:
                                pmt_sp_mask = self.pmt_sp_pruning_mask(
                                        data, pmt_sp_edges.edge_index, edge_distance.float())
                        else:
                                pmt_sp_mask = None

                        data["sp"].x = self.checkpoint(
                        self.pmt_to_nexus, (data["pmt"].x, data["sp"].x),
                        pmt_sp_edges.edge_index, pmt_sp_mask)
                else:
                        pmt_sp_mask = None

                # message-passing from pmt to flash
                data["flash"].x = self.checkpoint(
                        self.pmt_to_flash, (data["pmt"].x, data["flash"].x),
                        data["pmt", "in", "flash"].edge_index)

                # message-passing from flash to interaction
                data["evt"].x = self.checkpoint(
                        self.flash_to_interaction, (data["flash"].x, data["evt"].x),
                        data["flash", "in", "evt"].edge_index)

                # message-passing from interaction to flash
                data["flash"].x = self.checkpoint(
                        self.interaction_to_flash, (data["evt"].x, data["flash"].x),
                        data["flash", "in", "evt"].edge_index[(1,0), :])

                # message-passing from flash to pmt
                data["pmt"].x = self.checkpoint(
                        self.flash_to_pmt, (data["flash"].x, data["pmt"].x),
                        data["pmt", "in", "flash"].edge_index[(1,0), :])

                # message-passing from spacepoint to pmt
                if self.use_legacy_sp_pmt_edges:
                        data["pmt"].x = self.checkpoint(
                        self.nexus_to_pmt, (data["sp"].x, data["pmt"].x),
                        data["sp", "knn", "pmt"].edge_index)
                elif "edge_index" in pmt_sp_edges and pmt_sp_edges.edge_index.numel() > 0:
                        data["pmt"].x = self.checkpoint(
                        self.nexus_to_pmt, (data["sp"].x, data["pmt"].x),
                        pmt_sp_edges.edge_index[(1,0), :], pmt_sp_mask)

                # reverse message-passing from pmt to pmt
                if "edge_index" in pmt_edges and pmt_edges.edge_index.numel() > 0:
                        data["pmt"].x = self.checkpoint(
                        self.pmt_to_pmt, (data["pmt"].x, data["pmt"].x),
                        pmt_edges.edge_index[(1,0), :])

                # reverse message-passing from ophit to ophit
                if "edge_index" in ophit_edges and ophit_edges.edge_index.numel() > 0:
                        data["ophit"].x = self.checkpoint(
                        self.ophit_to_ophit, (data["ophit"].x, data["ophit"].x),
                        ophit_edges.edge_index[(1,0), :])

                # message-passing from pmt to ophit
                data["ophit"].x = self.checkpoint(
                        self.pmt_to_ophit, (data["pmt"].x, data["ophit"].x),
                        data["ophit", "in", "pmt"].edge_index[(1,0), :])
