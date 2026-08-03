"""NuGraph optical convolution module"""
import torch
import torch.nn.functional as F
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
                     sp_max_degree_init: float = 12.0,
                     sp_degree_temperature: float = 0.5,
                     use_pmt_sp_pruning: bool = True,
                     use_checkpointing: bool = True):
                super().__init__()

                self.use_checkpointing = use_checkpointing
                self.sp_degree_temperature = sp_degree_temperature
                self.use_pmt_sp_pruning = use_pmt_sp_pruning

                # learnable degree bound for spacepoints for pmt-sp edge pruning
                init = max(sp_max_degree_init - 1.0, 1e-3)
                self.sp_max_degree_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(init))))

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
                """Returns pruning mask for pmt-sp edges."""
                if edge_index.numel() == 0:
                        return torch.empty((0,), dtype=torch.float, device=edge_index.device)

                dst = edge_index[1].long()
                max_degree = 1.0 + F.softplus(self.sp_max_degree_param)
                max_degree_int = max(1, int(torch.floor(max_degree.detach()).item()))

                hard_mask = torch.zeros(edge_distance.size(0), dtype=torch.float, device=edge_distance.device)
                soft_mask = torch.zeros_like(hard_mask)

                for sp in dst.unique(sorted=True):
                        group_idx = (dst == sp).nonzero(as_tuple=False).squeeze(1)
                        if group_idx.numel() == 0:
                                continue
                        order = torch.argsort(edge_distance[group_idx], descending=False)
                        sorted_idx = group_idx[order]
                        rank = torch.arange(order.numel(), dtype=edge_distance.dtype, device=edge_distance.device)

                        hard_keep = (rank < max_degree_int).to(torch.float)
                        soft_keep = torch.sigmoid((max_degree - (rank + 0.5)) / self.sp_degree_temperature)

                        hard_mask[sorted_idx] = hard_keep
                        soft_mask[sorted_idx] = soft_keep

                mask = hard_mask + soft_mask - soft_mask.detach()

                if "sp" in data.node_types:
                        data["sp"].pmt_degree_cap = max_degree.detach().reshape(1)
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

                # message-passing from ophit to ophit within same pmt
                ophit_edges = data["ophit", "knn", "ophit"]
                # hotfix since some batches don't have these edges - shouldn't be the case
                if "edge_index" in ophit_edges and ophit_edges.edge_index.numel() > 0:
                        data["ophit"].x = self.checkpoint(
                        self.ophit_to_ophit, (data["ophit"].x, data["ophit"].x),
                        ophit_edges.edge_index)

                # message-passing from PMTs to PMTs
                pmt_edges = data["pmt", "knn", "pmt"]
                # same hotfix as above
                if "edge_index" in pmt_edges and pmt_edges.edge_index.numel() > 0:
                        data["pmt"].x = self.checkpoint(
                        self.pmt_to_pmt, (data["pmt"].x, data["pmt"].x),
                        pmt_edges.edge_index)

                # message-passing from pmt to spacepoint
                pmt_sp_edges = data["pmt", "knn", "sp"]
                # same hotfix as above
                if "edge_index" in pmt_sp_edges and pmt_sp_edges.edge_index.numel() > 0:
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
                if "edge_index" in pmt_sp_edges and pmt_sp_edges.edge_index.numel() > 0:
                        data["pmt"].x = self.checkpoint(
                        self.nexus_to_pmt, (data["sp"].x, data["pmt"].x),
                        pmt_sp_edges.edge_index[(1,0), :], pmt_sp_mask)

                # reverse message-passing from pmt to pmt
                if "edge_index" in pmt_edges and pmt_edges.edge_index.numel() > 0:
                        data["pmt"].x = self.checkpoint(
                        self.pmt_to_pmt, (data["pmt"].x, data["pmt"].x),
                        pmt_edges.edge_index[(1,0), :])

                # reverse message-passing from ophit to ophit within same pmt
                if "edge_index" in ophit_edges and ophit_edges.edge_index.numel() > 0:
                        data["ophit"].x = self.checkpoint(
                        self.ophit_to_ophit, (data["ophit"].x, data["ophit"].x),
                        ophit_edges.edge_index[(1,0), :])

                # message-passing from pmt to ophit
                data["ophit"].x = self.checkpoint(
                        self.pmt_to_ophit, (data["pmt"].x, data["ophit"].x),
                        data["ophit", "in", "pmt"].edge_index[(1,0), :])
