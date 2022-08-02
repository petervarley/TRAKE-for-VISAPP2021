#ifndef _CONSTANTS_HPP_
#define _CONSTANTS_HPP_

////////////////////////////////////////////////////////////////////////////////////////////

// These are geometry constants
constexpr float tcW = 1.0f;			// camera quality, 1.0 is a standard laptop webcam
constexpr float tcZ = 166.0f;		// ideal face width in pixels at a distance of 1m

// These are empirical observations from the LFW dataset
constexpr float tcF = 0.79f;		// ideal mouth location as proportion of face height, measured from top of face
constexpr float tcG = 0.41f;		// ideal width of mouth as proportion of face width
constexpr float tcH = 0.23f;		// ideal height of mouth as proportion of face height

constexpr float tcI = 0.5833f;		// ideal nose location as proportion of face height, measured from top of face
constexpr float tcJ = 0.30f;		// ideal width of nose as proportion of face width
constexpr float tcK = 0.25f;		// ideal height of nose as proportion of face height

constexpr float tcL = 0.35f;		// ideal eye location as proportion of face height, measured from top of face
constexpr float tcM = 0.23f;		// ideal width of eye as proportion of face width
constexpr float tcN = 0.23f;		// ideal height of eye as proportion of face height
constexpr float tcO = 0.30f;		// ideal horizontal distance between eye midpoints as proportion of face width

////////////////////////////////////////////////////////////////////////////////////////////

#endif
