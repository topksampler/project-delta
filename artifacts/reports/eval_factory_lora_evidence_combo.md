{
  "avg_content_score": 0.9157,
  "by_drift_type": {
    "added": 0.35,
    "changed": 0.0,
    "removed": 0.3333,
    "stable": 0.9402
  },
  "by_entity_type": {
    "cli_flag": 0.9824,
    "config_field": 0.8251,
    "env_var": 0.9476,
    "public_export": 0.881
  },
  "by_family_drift": {
    "cli_flag:added": {
      "accuracy": 0.3,
      "n": 5
    },
    "cli_flag:changed": {
      "accuracy": 0.0,
      "n": 1
    },
    "cli_flag:stable": {
      "accuracy": 1.0,
      "n": 249
    },
    "config_field:changed": {
      "accuracy": 0.0,
      "n": 7
    },
    "config_field:removed": {
      "accuracy": 0.0,
      "n": 1
    },
    "config_field:stable": {
      "accuracy": 0.8558,
      "n": 215
    },
    "env_var:added": {
      "accuracy": 0.4,
      "n": 5
    },
    "env_var:removed": {
      "accuracy": 0.5,
      "n": 2
    },
    "env_var:stable": {
      "accuracy": 0.9786,
      "n": 117
    },
    "public_export:stable": {
      "accuracy": 0.881,
      "n": 21
    }
  },
  "by_probe_form": {
    "version_delta": 0.7694,
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
  "exact_accuracy": 0.8604,
  "n_samples": 623,
  "schema": "delta.eval_factory.run_summary.v1"
}
