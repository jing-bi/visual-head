from transformers.generation.logits_process import LogitsProcessor
import torch


class ProbCFGLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        guidance_scale: float,
        use_log: bool = False,
    ):
        self.guidance_scale = guidance_scale
        self.use_log = use_log

    def __call__(self, input_ids, scores):
        if self.use_log:
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)

        uncond_logits, positive_logits, negative_logits = (
            scores[:1],
            scores[1:2],
            scores[2:3],
        )

        cond_logits = (
            uncond_logits
            + (
                self.guidance_scale * (positive_logits - uncond_logits)
                + self.guidance_scale * (uncond_logits - negative_logits)
            )
            / 2
        )
        # cond_logits = (
        #     self.guidance_scale * (cond_logits - uncond_logits) + uncond_logits
        # )

        # directly copy two.
        # logits = torch.cat([cond_logits, cond_logits, cond_logits], dim=0)
        logits = torch.cat([uncond_logits, uncond_logits, uncond_logits], dim=0)
        return logits.to("cuda:0")


class PassLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        guidance_scale: float,
        use_log: bool = False,  # whether to use log softmax.
    ):
        self.guidance_scale = guidance_scale
        self.use_log = use_log

    def __call__(self, input_ids, scores):

        return scores.to("cuda:0")
