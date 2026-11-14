// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const selectedRecord = vi.hoisted(() => ({
  cameras: [
    {
      id: "front-id",
      name: "front",
      device_id: "browser-device-id",
      type: "opencv",
      camera_index: 0,
      width: 640,
      height: 480,
      rotation: 90,
    },
  ],
}));

vi.mock("@/hooks/useRobots", () => ({
  useRobots: () => ({ selectedRecord, isLoading: false }),
}));
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://lelab.test" }),
}));

import TeleopCameraPanel from "./TeleopCameraPanel";

let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div");
  root = createRoot(container);
});

afterEach(() => act(() => root.unmount()));

test("teleoperation preview automatically uses LeLab camera stream instead of browser capture", () => {
  act(() => root.render(<TeleopCameraPanel />));

  expect(container.textContent).not.toContain("Browser preview is disabled");
  expect(container.querySelector("video")).toBeNull();

  const image = container.querySelector("img")!;
  expect(image.src).toBe("http://lelab.test/teleoperation-camera-feed/front?k=0");
  expect(image.alt).toBe("front live feed");
  expect(image.parentElement!.style.aspectRatio).toBe("480 / 640");

  act(() => image.dispatchEvent(new Event("error")));
  expect(container.textContent).toContain(
    "Camera feed unavailable. Check the LeLab server logs, then retry.",
  );
});
