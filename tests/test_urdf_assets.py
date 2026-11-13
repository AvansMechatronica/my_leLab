from pathlib import Path
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree


def test_urdf_meshes_resolve_from_urdf_url() -> None:
    public = Path(__file__).resolve().parents[1] / "frontend" / "public"
    urdf_url = "http://localhost/ur_urdf/urdf/ur.urdf"
    urdf = public / urlsplit(urdf_url).path.lstrip("/")
    meshes = ElementTree.parse(urdf).findall(".//mesh")

    assert len(meshes) == 32
    for mesh in meshes:
        filename = mesh.attrib["filename"]
        url = urljoin(urdf_url, filename)
        assert (public / urlsplit(url).path.lstrip("/")).is_file(), url
