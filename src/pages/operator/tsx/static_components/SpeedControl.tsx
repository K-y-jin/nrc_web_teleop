import React from "react";
import "operator/css/SpeedControl.css";

/**Details of a velocity setting */
type VelocityDetails = {
    /**Name of the setting to display on the button */
    label: string;
    /**The speed of this setting */
    scale: number;
};

/**Props for {@link SpeedControl} */
type SpeedControlProps = {
    /** Initial speed when interface first loaded. */
    scale: number;
    /**
     * Callback function when a new speed is selected.
     * @param newSpeed the new selected speed
     */
    onChange: (newScale: number) => void;
};

/**
export const VELOCITY_SCALE: VelocityDetails[] = [
    { label: "Slowest", scale: 0.2 },
    { label: "Slow", scale: 0.4 },
    { label: "Medium", scale: 0.8 },
    { label: "Fast", scale: 1.2 },
    { label: "Fastest", scale: 1.6 },
];
**/
/** Revised Settings **/
export const VELOCITY_SCALE: VelocityDetails[] = [
    { label: "Slowest", scale: 0.8 },
    { label: "Slow", scale: 1.6 },
    { label: "Medium", scale: 3.2 },
    { label: "Fast", scale: 4.8 },
    { label: "Fastest", scale: 5.6 },
];

/**The speed the interface should initialize with */
export const DEFAULT_VELOCITY_SCALE: number = VELOCITY_SCALE[2].scale;

/**
 * Set of buttons so the user can control the scaling of the speed for all controls.
 * @param props see {@link SpeedControlProps}
 */
export const SpeedControl = (props: SpeedControlProps) => {
    /** Maps the velocity labels and speeds to radio buttons */
    const mapFunc = ({ scale, label }: VelocityDetails) => {
        const active = scale === props.scale;
        return (
            <button
                key={label}
                className={active ? "btn-blue font-white" : ""}
                onClick={() => props.onChange(scale)}
            >
                {label}
            </button>
        );
    };

    return (
        <div id="velocity-control-container">{VELOCITY_SCALE.map(mapFunc)}</div>
    );
};
