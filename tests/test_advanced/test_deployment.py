"""Tests for the model-serving request handler."""

from __future__ import annotations

import torch
from torch import nn

from pytorch_mastery_hub.advanced.deployment import ModelServer, predict_response


def _server() -> ModelServer:
    torch.manual_seed(0)
    return ModelServer(nn.Linear(3, 2), preprocessing_fn=torch.tensor)


class TestPredictResponse:
    def test_success_is_json_serialisable(self):
        body, status = predict_response(_server(), {"input": [[1.0, 2.0, 3.0]]})
        assert status == 200
        assert body["status"] == "success"
        assert isinstance(body["prediction"], list)
        assert len(body["prediction"][0]) == 2

    def test_malformed_body_is_400(self):
        for payload in (None, [], {"x": 1}):
            body, status = predict_response(_server(), payload)
            assert status == 400
            assert "input" in body["error"]

    def test_model_error_does_not_leak_exception_text(self):
        # Wrong feature count makes the Linear layer raise a RuntimeError whose
        # message includes tensor shapes; none of that may reach the client.
        body, status = predict_response(_server(), {"input": [[1.0, 2.0]]})
        assert status == 500
        assert body == {"error": "prediction failed", "status": "error", "request_id": 1}

    def test_unexpected_exception_does_not_leak(self):
        class Boom(ModelServer):
            def predict(self, input_data):
                raise RuntimeError("/secret/path/model.pt corrupted")

        body, status = predict_response(Boom(nn.Linear(1, 1)), {"input": [0.0]})
        assert status == 500
        assert "secret" not in str(body)
