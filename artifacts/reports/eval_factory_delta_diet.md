{
  "avg_content_score": 0.8274,
  "by_drift_type": {
    "added": 0.45,
    "changed": 0.375,
    "removed": 0.5,
    "stable": 0.8414
  },
  "by_entity_type": {
    "cli_flag": 0.8294,
    "config_field": 0.8094,
    "env_var": 0.8669,
    "public_export": 0.7619
  },
  "by_family_drift": {
    "cli_flag:added": {
      "accuracy": 0.4,
      "n": 5
    },
    "cli_flag:changed": {
      "accuracy": 0.0,
      "n": 1
    },
    "cli_flag:stable": {
      "accuracy": 0.8414,
      "n": 249
    },
    "config_field:changed": {
      "accuracy": 0.4286,
      "n": 7
    },
    "config_field:removed": {
      "accuracy": 0.5,
      "n": 1
    },
    "config_field:stable": {
      "accuracy": 0.8233,
      "n": 215
    },
    "env_var:added": {
      "accuracy": 0.5,
      "n": 5
    },
    "env_var:removed": {
      "accuracy": 0.5,
      "n": 2
    },
    "env_var:stable": {
      "accuracy": 0.8889,
      "n": 117
    },
    "public_export:stable": {
      "accuracy": 0.7619,
      "n": 21
    }
  },
  "by_probe_form": {
    "version_delta": 0.5183,
    "versioned_existence": 0.995
  },
  "counts": {
    "by_drift_type": {
      "added": 10,
      "changed": 8,
      "removed": 3,
      "stable": 602
    },
    "by_entity_type": {
      "cli_flag": 255,
      "config_field": 223,
      "env_var": 124,
      "public_export": 21
    }
  },
  "exact_accuracy": 0.8154,
  "n_samples": 623,
  "schema": "delta.eval_factory.run_summary.v1"
}
