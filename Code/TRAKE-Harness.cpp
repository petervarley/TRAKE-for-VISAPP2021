#include <windows.h>
#include <gdiplus.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <math.h>

#include "CVFrame.hpp"
#include "HaarCascades.hpp"
#include "Noses.hpp"
#include "EyeRegions.hpp"
#include "Mouths.hpp"
#include "Pupils.hpp"
#include "Constants.hpp"
#include "Known.hpp"
#include "FileLocations.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

static WNDCLASSEX wc; /* A properties struct of our window */
static HWND hwnd; /* A 'HANDLE', hence the H, or a pointer to our window */
static HMENU hmenu;
static HMENU hmenutbl;
static HMENU hmenueva;
static HMENU hmenucas;
static HMENU hmenusho;
static HMENU hmenuxit;

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

static bool is_an_image (char *Filename)
{
	char *suffix = strrchr(Filename,'.');
	if (!suffix)  return false;
	if (!strcmp(suffix,".jpg"))  return true;
	if (!strcmp(suffix,".png"))  return true;
	if (!strcmp(suffix,".JPG"))  return true;
	if (!strcmp(suffix,".PNG"))  return true;
	return false;
}

//--------------------------------------------------------------------------------------------

static void process_source_images (const std::string &DirectoryName, char *options)
{
	qUseKnown = qUseKnownFaces || qUseKnownEyes || qUseKnownNoses || qUseKnownMouths || qUseKnownPupils;
	if (qUseKnown)  initialise_known();
	if (qUseCascades && !qHaveCascades)  load_haar_cascades();
	qShowFaceWindow = true;

	HANDLE hFind;
	WIN32_FIND_DATA ffd;
	char lpFileName[MAX_PATH];
	sprintf(lpFileName,"%s/*",DirectoryName.c_str());

	hFind = FindFirstFile(lpFileName, &ffd);

	setup_source_window();

	do
	{
		if (is_an_image(ffd.cFileName))  read_and_display_frame(DirectoryName.c_str(),ffd.cFileName,flocOutput,options);
	}
	while (FindNextFile(hFind, &ffd) != 0);

	FindClose(hFind);
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

void paint_main_window (void)
{
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint( hwnd, &ps );

	COLORREF RedColour = RGB(160,0,0);
	SelectObject(ps.hdc, GetStockObject(DC_BRUSH));
	SetDCBrushColor(hdc,RedColour);
	Rectangle(hdc,   0,  0, 340,180);

	COLORREF BlueColour = RGB(0,0,144);
	SelectObject(ps.hdc, GetStockObject(DC_BRUSH));
	SetDCBrushColor(hdc,BlueColour);
	Rectangle(hdc,1020,  0,1360,180);

	COLORREF GreenColour = RGB(0,128,0);
	SelectObject(ps.hdc, GetStockObject(DC_BRUSH));
	SetDCBrushColor(hdc,GreenColour);
	Rectangle(hdc,   0,480, 340,660);

	COLORREF YellowColour = RGB(144,128,0);
	SelectObject(ps.hdc, GetStockObject(DC_BRUSH));
	SetDCBrushColor(hdc,YellowColour);
	Rectangle(hdc,1020,480,1360,660);

	EndPaint(hwnd, &ps);
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

#define IDM_VISAPP_TABLE_1      1301

#define IDM_VISAPP_TABLE_3      1503
#define IDM_VISAPP_TABLE_4      1504
#define IDM_VISAPP_TABLE_5      1505
#define IDM_VISAPP_TABLE_6      1506
#define IDM_VISAPP_TABLE_7      1507
#define IDM_VISAPP_COMBINE_CT   1520
#define IDM_VISAPP_COMBINE_BD   1522
#define IDM_VISAPP_COMBINE_CTU  1523
#define IDM_VISAPP_COMBINE_VHTU 1524

#define IDM_FACE_CASCADE_0      1600
#define IDM_FACE_CASCADE_1      1601
#define IDM_FACE_CASCADE_2      1602

#define IDM_SHO_NONE            1700
#define IDM_SHO_EYES            1710
#define IDM_SHO_NOSE            1720
#define IDM_SHO_FAST            1790
#define IDM_SHO_SLOW            1799

#define IDM_EXIT                2000

///////////////////////////////////////////////////////////////////////////////////////////////

/* This is where all the input to the window goes to */
LRESULT CALLBACK WndProc(HWND hwnd, UINT Message, WPARAM wParam, LPARAM lParam)
{
	switch(Message)
	{
		case WM_COMMAND:
		{
			switch(LOWORD(wParam))
			{
				case IDM_VISAPP_TABLE_1:
					qWhichNoseCascade = NOSE_MCS;
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_VISAPP_TABLE_3:
					qWhichNoseCascade = NOSE_FRONTAL_19x13;
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_VISAPP_TABLE_4:
					qWhichNoseCascade = NOSE_FRONTAL_25x17;
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_VISAPP_TABLE_5:
					qWhichNoseCascade = NOSE_FRONTAL_31x21;
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_VISAPP_TABLE_6:
					qWhichNoseCascade = NOSE_PROFILE_ONLY;
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_VISAPP_TABLE_7:
				case IDM_VISAPP_COMBINE_CT:	// as this is the default anyway
					qWhichNoseCascade = 2;
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_VISAPP_COMBINE_BD:
					qWhichNoseCascade = 2;
					nose_fern_list = "BD";
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_VISAPP_COMBINE_CTU:
					qWhichNoseCascade = 2;
					nose_fern_list = "CTU";
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_VISAPP_COMBINE_VHTU:
					qWhichNoseCascade = 2;
					nose_fern_list = "VHTU";
					qUseKnownFaces = true;
					process_source_images(flocFacesFemale," ");
					process_source_images(flocFacesMale," ");
				break;

				case IDM_FACE_CASCADE_0:
					qWhichFaceCascade = 0;
				break;

				case IDM_FACE_CASCADE_1:
					qWhichFaceCascade = 1;
				break;

				case IDM_FACE_CASCADE_2:
					qWhichFaceCascade = 2;
				break;

				case IDM_SHO_NONE:
					qShowEyeWindows = qShowNoseWindow = false;
				break;

				case IDM_SHO_EYES:
					qShowEyeWindows = true;
				break;

				case IDM_SHO_NOSE:
					qShowNoseWindow = true;
				break;

				case IDM_SHO_FAST:
					qFrameSpeed = 100;
				break;

				case IDM_SHO_SLOW:
					qFrameSpeed = 2000;
				break;

				case IDM_EXIT:
					PostQuitMessage(0);
				break;

				default:
				break;
          	}

			break;
		}

		case WM_PAINT:
		{
			paint_main_window();
			break;
		}

		case WM_KEYDOWN:
		{
			switch(LOWORD(wParam))
			{
				case 'Z':
				case 'z':
					InvalidateRect(hwnd,NULL,true);
				break;

				case 'M':
				case 'm':
					InvalidateRect(hwnd,NULL,true);
				break;
			}
			break;
		}

		/* Upon destruction, tell the main thread to stop */
		case WM_DESTROY:
		{
			PostQuitMessage(0);
			break;
		}

		/* All other messages (a lot of them) are processed using default procedures */
		default:
			return DefWindowProc(hwnd, Message, wParam, lParam);
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////

/* The 'main' function of Win32 GUI programs: this is where execution starts */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	std::ofstream ofs("TRAKEHarnessLog.txt");
	std::clog.rdbuf(ofs.rdbuf());
 	std::clog <<  "TRAKE Eye Design Test Harness" << std::endl;

	MSG msg; /* A temporary location for all messages */
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR           gdiplusToken;
	Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

	/* zero out the struct and set the stuff we want to modify */
	memset(&wc,0,sizeof(wc));
	wc.cbSize		 = sizeof(WNDCLASSEX);
	wc.lpfnWndProc	 = WndProc; /* This is where we will send messages to */
	wc.hInstance	 = hInstance;
	wc.hCursor		 = LoadCursor(NULL, IDC_ARROW);

	/* White, COLOR_WINDOW is just a #define for a system color, try Ctrl+Clicking it */
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
	wc.lpszClassName = "WindowClass";
	wc.hIcon		 = LoadIcon(NULL, IDI_APPLICATION); /* Load a standard icon */
	wc.hIconSm		 = LoadIcon(NULL, IDI_APPLICATION); /* use the name "A" to use the project icon */

	if(!RegisterClassEx(&wc))
	{
		MessageBox(NULL, "Window Registration Failed!","Error!",MB_ICONEXCLAMATION|MB_OK);
		return 0;
	}

	hmenu = CreateMenu();

	if(hmenu == NULL)
	{
		MessageBox(NULL, "Menu Creation Failed!","Error!",MB_ICONEXCLAMATION|MB_OK);
		return 0;
	}

	hmenutbl = CreatePopupMenu();
	hmenueva = CreatePopupMenu();
	hmenucas = CreatePopupMenu();
	hmenusho = CreatePopupMenu();
	hmenuxit = CreatePopupMenu();

    AppendMenu(hmenu, MF_STRING | MF_POPUP, (UINT) hmenutbl, "&VISAPP Tables");
    AppendMenu(hmenu, MF_STRING | MF_POPUP, (UINT) hmenueva, "&VISAPP Evaluations");
    AppendMenu(hmenu, MF_STRING | MF_POPUP, (UINT) hmenucas, "&Choose Face Cascade");
    AppendMenu(hmenu, MF_STRING | MF_POPUP, (UINT) hmenusho, "&Show Subwindows");
    AppendMenu(hmenu, MF_STRING | MF_POPUP, (UINT) hmenuxit, "&Exit");

    AppendMenu(hmenutbl, MF_STRING, (UINT) IDM_VISAPP_TABLE_1, "&VISAPP Table 1");
    AppendMenu(hmenutbl, MF_STRING, (UINT) IDM_VISAPP_TABLE_3, "&VISAPP Table 3");
    AppendMenu(hmenutbl, MF_STRING, (UINT) IDM_VISAPP_TABLE_4, "&VISAPP Table 4");
    AppendMenu(hmenutbl, MF_STRING, (UINT) IDM_VISAPP_TABLE_5, "&VISAPP Table 5");
    AppendMenu(hmenutbl, MF_STRING, (UINT) IDM_VISAPP_TABLE_6, "&VISAPP Table 6");
    AppendMenu(hmenutbl, MF_STRING, (UINT) IDM_VISAPP_TABLE_7, "&VISAPP Table 7");

    AppendMenu(hmenueva, MF_STRING, (UINT) IDM_VISAPP_COMBINE_CT, "&VISAPP Centre and Triangular");
    AppendMenu(hmenueva, MF_STRING, (UINT) IDM_VISAPP_COMBINE_CTU, "&VISAPP Centre and both Triangulars");
    AppendMenu(hmenueva, MF_STRING, (UINT) IDM_VISAPP_COMBINE_BD, "&VISAPP Single and Double Ferns");
    AppendMenu(hmenueva, MF_STRING, (UINT) IDM_VISAPP_COMBINE_VHTU, "&VISAPP Triple Ferns");

    AppendMenu(hmenucas, MF_STRING, (UINT) IDM_FACE_CASCADE_0, "&Use Face Cascade 0");
    AppendMenu(hmenucas, MF_STRING, (UINT) IDM_FACE_CASCADE_1, "&Use Face Cascade 1");
    AppendMenu(hmenucas, MF_STRING, (UINT) IDM_FACE_CASCADE_2, "&Use Face Cascade 2");

    AppendMenu(hmenusho, MF_STRING, (UINT) IDM_SHO_NONE, "&None");
    AppendMenu(hmenusho, MF_STRING, (UINT) IDM_SHO_EYES, "&Eyes");
    AppendMenu(hmenusho, MF_STRING, (UINT) IDM_SHO_NOSE, "&Nose");
    AppendMenu(hmenusho, MF_STRING, (UINT) IDM_SHO_FAST, "&FAST");
    AppendMenu(hmenusho, MF_STRING, (UINT) IDM_SHO_SLOW, "&SLOW");

    AppendMenu(hmenuxit, MF_STRING, (UINT) IDM_EXIT, "&Exit");


	hwnd = CreateWindowEx(WS_EX_CLIENTEDGE,"WindowClass","TRAKE",WS_VISIBLE|WS_OVERLAPPEDWINDOW,
		0, /* x */
		0, /* y */
		1360, /* width */
		 720, /* height */
		NULL,NULL,hInstance,NULL);

	if(hwnd == NULL)
	{
		MessageBox(NULL, "Window Creation Failed!","Error!",MB_ICONEXCLAMATION|MB_OK);
		return 0;
	}

	//SetWindowPos(hwnd,HWND_TOP,0,0,1080,640,SWP_SWP_SHOWWINDOW);

	SetMenu(hwnd,hmenu);
	DrawMenuBar(hwnd);

	/*
		This is the heart of our program where all input is processed and
		sent to WndProc. Note that GetMessage blocks code flow until it receives something, so
		this loop will not produce unreasonably high CPU usage
	*/
	while(GetMessage(&msg, NULL, 0, 0) > 0)
	{ /* If no error is received... */
		TranslateMessage(&msg); /* Translate key codes to chars if present */
		DispatchMessage(&msg); /* Send it to WndProc */
	}

	Gdiplus::GdiplusShutdown(gdiplusToken);
	return msg.wParam;
}

///////////////////////////////////////////////////////////////////////////////////////////////
