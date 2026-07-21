"""No-op implementations for ComfyUI subsystems omitted from ProjectSwift.

The ProjectSwift worker is API-only and never enables ComfyUI's asset catalog.
These shims avoid importing its database stack while preserving the prompt and
image HTTP APIs used by ProjectSwift.
"""


class _DisabledAssetSeeder:
    def disable(self):
        return None

    def is_disabled(self):
        return True

    def start(self, *args, **kwargs):
        return False

    def pause(self):
        return None

    def resume(self):
        return None

    def enqueue_enrich(self, *args, **kwargs):
        return None

    def shutdown(self):
        return None


asset_seeder = _DisabledAssetSeeder()


def register_assets_routes(*args, **kwargs):
    return None


def register_output_files(*args, **kwargs):
    return []


def register_file_in_place(*args, **kwargs):
    raise RuntimeError("The ProjectSwift lean runtime does not include asset indexing")


def get_known_subfolder_tags(*args, **kwargs):
    return []


def resolve_hash_to_path(*args, **kwargs):
    return None
