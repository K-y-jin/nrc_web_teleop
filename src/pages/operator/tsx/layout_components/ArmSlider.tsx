import React from "react";
import { JOINT_LIMITS, ValidJoints, RobotPose } from "shared/util";
import { FunctionProvider } from "../function_providers/FunctionProvider";
import "operator/css/ArmSlider.css";

const RAD2DEG = 180 / Math.PI;

type SliderProps = {
    jointName: ValidJoints;
    label: string;
    robotPose?: RobotPose;
    disabled: boolean;
    unit?: "m" | "deg" | "norm";
    centerZero?: boolean;
};

function formatValue(value: number, unit: "m" | "deg" | "norm", min?: number, max?: number): string {
    if (unit === "deg") {
        return `${(value * RAD2DEG).toFixed(1)}\u00B0`;
    }
    if (unit === "norm" && min !== undefined && max !== undefined) {
        const normalized = (value - min) / (max - min);
        return normalized.toFixed(2);
    }
    return `${value.toFixed(2)} m`;
}

function formatLimit(value: number, unit: "m" | "deg" | "norm", min?: number, max?: number): string {
    if (unit === "deg") {
        return `${(value * RAD2DEG).toFixed(0)}\u00B0`;
    }
    if (unit === "norm" && min !== undefined && max !== undefined) {
        const normalized = (value - min) / (max - min);
        return normalized.toFixed(1);
    }
    return `${value.toFixed(2)}`;
}

/** Vertical slider — used for Lift (H) and Pitch */
export const LiftSlider = (props: SliderProps) => {
    const { jointName, label, robotPose, disabled, unit = "m", centerZero = false } = props;
    const limits = JOINT_LIMITS[jointName]!;
    const min = limits[0];
    const max = limits[1];
    const currentPosition = robotPose?.[jointName] ?? (centerZero ? 0 : min);
    const [targetPosition, setTargetPosition] = React.useState<number | null>(
        null,
    );

    const toRatio = (v: number) => Math.max(0, Math.min(1, (v - min) / (max - min)));
    const positionRatio = toRatio(currentPosition);
    const zeroRatio = toRatio(0);

    const handleClick = (e: React.PointerEvent<HTMLDivElement>) => {
        if (disabled) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const ratio = 1 - (e.clientY - rect.top) / rect.height;
        const clampedRatio = Math.max(0, Math.min(1, ratio));
        const target = min + clampedRatio * (max - min);
        setTargetPosition(target);
        FunctionProvider.moveToJointPosition(jointName, target);
    };

    const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
        if (e.buttons !== 1 || disabled) return;
        handleClick(e);
    };

    React.useEffect(() => {
        if (
            targetPosition !== null &&
            Math.abs(currentPosition - targetPosition) < 0.01
        ) {
            setTargetPosition(null);
        }
    }, [currentPosition, targetPosition]);

    const fillStyle = centerZero
        ? {
              bottom: `${Math.min(positionRatio, zeroRatio) * 100}%`,
              height: `${Math.abs(positionRatio - zeroRatio) * 100}%`,
          }
        : { height: `${positionRatio * 100}%` };

    return (
        <div className={`arm-slider-vertical${disabled ? " disabled" : ""}`}>
            <div className="arm-slider-label">{label}</div>
            <div className="arm-slider-limit arm-slider-limit-max">
                {formatLimit(max, unit, min, max)}
            </div>
            <div
                className="arm-slider-track-v"
                onPointerDown={handleClick}
                onPointerMove={handlePointerMove}
            >
                <div
                    className={centerZero ? "arm-slider-fill-v-center" : "arm-slider-fill-v"}
                    style={fillStyle}
                />
                {centerZero && (
                    <div
                        className="arm-slider-zero-v"
                        style={{ bottom: `${zeroRatio * 100}%` }}
                    />
                )}
                <div
                    className="arm-slider-thumb-v"
                    style={{ bottom: `${positionRatio * 100}%` }}
                />
                {targetPosition !== null && (
                    <div
                        className="arm-slider-target-v"
                        style={{
                            bottom: `${toRatio(targetPosition) * 100}%`,
                        }}
                    />
                )}
            </div>
            <div className="arm-slider-limit arm-slider-limit-min">
                {formatLimit(min, unit, min, max)}
            </div>
            <div className="arm-slider-value">
                {formatValue(currentPosition, unit, min, max)}
            </div>
        </div>
    );
};

/** Horizontal slider — used for Extension (L), Yaw, Grip */
export const ExtensionSlider = (props: SliderProps) => {
    const { jointName, label, robotPose, disabled, unit = "m", centerZero = false } = props;
    const limits = JOINT_LIMITS[jointName]!;
    const min = limits[0];
    const max = limits[1];
    const currentPosition = robotPose?.[jointName] ?? (centerZero ? 0 : min);
    const [targetPosition, setTargetPosition] = React.useState<number | null>(
        null,
    );

    const toRatio = (v: number) => Math.max(0, Math.min(1, (v - min) / (max - min)));
    const positionRatio = toRatio(currentPosition);
    const zeroRatio = toRatio(0);

    const handleClick = (e: React.PointerEvent<HTMLDivElement>) => {
        if (disabled) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const ratio = (e.clientX - rect.left) / rect.width;
        const clampedRatio = Math.max(0, Math.min(1, ratio));
        const target = min + clampedRatio * (max - min);
        setTargetPosition(target);
        FunctionProvider.moveToJointPosition(jointName, target);
    };

    const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
        if (e.buttons !== 1 || disabled) return;
        handleClick(e);
    };

    React.useEffect(() => {
        if (
            targetPosition !== null &&
            Math.abs(currentPosition - targetPosition) < 0.01
        ) {
            setTargetPosition(null);
        }
    }, [currentPosition, targetPosition]);

    const fillStyle = centerZero
        ? {
              left: `${Math.min(positionRatio, zeroRatio) * 100}%`,
              width: `${Math.abs(positionRatio - zeroRatio) * 100}%`,
          }
        : { width: `${positionRatio * 100}%` };

    return (
        <div className={`arm-slider-horizontal${disabled ? " disabled" : ""}`}>
            <div className="arm-slider-h-track-row">
                <div className="arm-slider-limit arm-slider-limit-min-h">
                    {formatLimit(min, unit, min, max)}
                </div>
                <div
                    className="arm-slider-track-h"
                    onPointerDown={handleClick}
                    onPointerMove={handlePointerMove}
                >
                    <div
                        className={centerZero ? "arm-slider-fill-h-center" : "arm-slider-fill-h"}
                        style={fillStyle}
                    />
                    {centerZero && (
                        <div
                            className="arm-slider-zero-h"
                            style={{ left: `${zeroRatio * 100}%` }}
                        />
                    )}
                    <div
                        className="arm-slider-thumb-h"
                        style={{ left: `${positionRatio * 100}%` }}
                    />
                    {targetPosition !== null && (
                        <div
                            className="arm-slider-target-h"
                            style={{
                                left: `${toRatio(targetPosition) * 100}%`,
                            }}
                        />
                    )}
                </div>
                <div className="arm-slider-limit arm-slider-limit-max-h">
                    {formatLimit(max, unit, min, max)}
                </div>
            </div>
            <div className="arm-slider-h-info">
                <div className="arm-slider-label">{label}</div>
                <div className="arm-slider-value">
                    {formatValue(currentPosition, unit, min, max)}
                </div>
            </div>
        </div>
    );
};
