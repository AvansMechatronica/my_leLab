import React, { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useRobots } from "@/hooks/useRobots";
import { useApi } from "@/contexts/ApiContext";

/**
 * Optional live camera panel for the teleoperation page. Off by default so we
 * only request camera streams when the user asks to see them. LeLab owns each
 * camera connection and serves frames over MJPEG, avoiding a second browser
 * connection to a device already open on the robot host.
 *
 * A strict mirror of the selected robot's configured cameras: one live feed per
 * camera on the robot record (e.g. "wrist_cam", "webcam"), stacked vertically.
 * If the robot has none configured it shows nothing — teleop never surfaces a
 * device that wasn't deliberately added to the robot.
 */
const TeleopCameraPanel: React.FC = () => {
  const [enabled, setEnabled] = useState(true);
  // Bumped by the retry button to reconnect the MJPEG feeds.
  const [reloadKey, setReloadKey] = useState(0);
  const [failedFeeds, setFailedFeeds] = useState<Set<string>>(new Set());
  const { selectedRecord, isLoading: robotsLoading } = useRobots();
  const { baseUrl } = useApi();

  const configured = selectedRecord?.cameras ?? [];

  const retryFeeds = () => {
    setFailedFeeds(new Set());
    setReloadKey((key) => key + 1);
  };

  return (
    <div className="bg-gray-900 rounded-lg p-4 flex flex-col gap-4 h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-medium text-gray-200">Cameras</h2>
        <div className="flex items-center gap-2">
          {enabled && configured.length > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={retryFeeds}
              className="h-9 w-9 text-gray-400 hover:text-white flex-shrink-0"
              title="Retry camera feeds (e.g. after reconnecting a camera)"
              aria-label="Retry camera feeds"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          )}
          <Label htmlFor="teleop-camera-toggle" className="text-sm text-gray-400">
            {enabled ? "On" : "Off"}
          </Label>
          <Switch
            id="teleop-camera-toggle"
            checked={enabled}
            onCheckedChange={setEnabled}
          />
        </div>
      </div>

      {enabled ? (
        configured.length > 0 ? (
          <div className="flex flex-col gap-3 overflow-y-auto">
            {configured.map((camera) => {
              const quarterTurn = camera.rotation === 90 || camera.rotation === 270;
              const width = quarterTurn ? camera.height : camera.width;
              const height = quarterTurn ? camera.width : camera.height;
              const src =
                `${baseUrl}/teleoperation-camera-feed/${encodeURIComponent(camera.name)}` +
                `?k=${reloadKey}`;

              const failed = failedFeeds.has(camera.name);
              return (
                <div
                  key={`${camera.id}:${reloadKey}`}
                  className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden"
                >
                  <div style={{ aspectRatio: `${width} / ${height}` }}>
                    {failed ? (
                      <div className="w-full h-full flex items-center justify-center text-sm text-gray-400">
                        Camera feed unavailable. Check the LeLab server logs, then retry.
                      </div>
                    ) : (
                      <img
                        src={src}
                        alt={`${camera.name} live feed`}
                        className="w-full h-full object-contain"
                        onError={() =>
                          setFailedFeeds((current) => new Set(current).add(camera.name))
                        }
                      />
                    )}
                  </div>
                  <div className="p-2 text-sm text-gray-300 truncate border-t border-gray-800">
                    {camera.name}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-gray-500">
            {robotsLoading
              ? "Loading robot..."
              : "No cameras configured for this robot. Add them during calibration to see live feeds here."}
          </p>
        )
      ) : (
        <p className="text-sm text-gray-500">
          Turn on to watch your cameras while you teleoperate.
        </p>
      )}
    </div>
  );
};

export default TeleopCameraPanel;
