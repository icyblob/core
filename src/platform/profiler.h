#pragma once

#define ENABLE_PROFILER 1

#if ENABLE_PROFILER

#include "intrin.h"
#include "common_types.h"
#include "console_logging.h"
#include "public_settings.h"

#define WRITE_PROFILE_MESSAGES_TO_FILE 1

static constexpr unsigned int PROFILE_NAME_WIDTH = 64;
static constexpr unsigned int MAX_PROFILE_MODULE = 32;
static char profileLock[MAX_PROFILE_MODULE];
static char profileAllLock = 0;

#if WRITE_PROFILE_MESSAGES_TO_FILE
static CHAR16 profileFileName[64];
static CHAR16 timeStampPrefix[16];
#endif

struct ProfileTimer;

static ProfileTimer* gActiveTimer = NULL;

bool CheckTextMatch(const CHAR16* A, const CHAR16* B)
{
    // Iterate through each character in both strings
    while ((*A != 0) && (*B != 0))
    {
        // If characters do not match, return false
        if (*A != *B)
        {
            return false;
        }

        // Move to the next character
        A++;
        B++;
    }

    // If both strings ended, they match
    return (*A == 0) && (*B == 0);
}

struct ProfileTimer
{
public:
    ProfileTimer()
    {
        for (int i = 0; i < MAX_PROFILE_MODULE; i++)
        {
            mpChildren[i] = NULL;
        }
        setMem(mName, sizeof(mName), 0);
    }

    void Init(const CHAR16* name, const int level)
    {
        mLevel = level;
        mStart = 0;
        mElapsedTime = 0;
        mCallCount = 0;
        mParentIndex = -1;
        setText(mName, name);
    }

    void ResetTimer()
    {
        mStart = 0;
        mElapsedTime = 0;
    }

    void Start()
    {
        mIsActive = true;
        mCallCount++;
        mStart = __rdtsc();
    }

    void Stop()
    {
        mElapsedTime += __rdtsc() - mStart;

        // Set this zero here to detect if the stop is call without start
        mStart = 0;
        // Finish record time
        mIsActive = false;
    }


    // Get current level of this profile
    int GetLevel() const
    {
        return mLevel;
    }

    // Get avg number of clocks
    unsigned long long GetAvgTime() const
    {
        if (mCallCount > 0)
        {
            return mElapsedTime / mCallCount;
        }
        return 0;
    }

    // Get total calls
    unsigned long long GetCallCount()  const
    {
        return mCallCount;
    }

    // Get total calls
    bool isActive()
    {
        return mIsActive;
    }

    void SetParentIdx(const int parentIdx)
    {
        mParentIndex = parentIdx;
    }

    int getParentIdx()  const
    {
       return mParentIndex;
    }

    // Add a module as children
    void AddChildren(ProfileTimer* pChild)
    {
        bool foundChild = false;
        for (int i = 0; i < mChildrenCount; i++)
        {
            if (pChild == mpChildren[i])
            {
                foundChild = true;
                break;
            }
        }
        if (!foundChild)
        {
            mpChildren[mChildrenCount] = pChild;
            mChildrenCount++;
        }
    }

    // Name of this time record
    CHAR16 mName[32];

    int mLevel = 0;

    // Childen of this Profiler
    ProfileTimer* mpChildren[MAX_PROFILE_MODULE];
    int mChildrenCount = 0;

private:

    // Elapsed time in clocks
    unsigned long long mElapsedTime = 0;

    // Number of times called
    unsigned long long mCallCount = 0;

    // Elapsed time in clocks
    unsigned long long mStart = 0;

    // Flag indicate this profiler is still active
    bool mIsActive = false;

    // Parent of this Profiler
    ProfileTimer* mpParent = NULL;
    int mParentIndex = -1;
};

// Simple profiler. This is not a thread safe struct, make sure the profile is called from the same thread
struct Profiler
{
public:
    Profiler()
    {
        mCurrentProfile = -1;
        mCurrentLevel = 0;
        nModules = 0;
        setMem(printLog, sizeof(printLog), 0);
    }

    void Profile(const CHAR16* name)
    {
        // Check the capacity of the profile
        if (nModules > MAX_PROFILE_MODULE)
        {
            return;
        }
        // Check if the current active profile is still actived
        int parentProfile = -1;
        if (mCurrentProfile >= 0 && mProfiles[mCurrentProfile].isActive())
        {
            parentProfile = mCurrentProfile;
            mCurrentLevel = mProfiles[parentProfile].GetLevel() + 1;
        }

        // Check if there is existed modules
        int foundModule = -1;
        for (int i = 0; i < nModules; i++)
        {
            // Look for matching name
            if (CheckTextMatch(mProfiles[i].mName, name))
            {
                // Verify if the level calling is the same
                if (mProfiles[i].mLevel == mCurrentLevel)
                {
                    // Verify the parnet level if there is any
                    if (parentProfile == -1 || mProfiles[i].getParentIdx() == parentProfile)
                    {
                        foundModule = i;
                        break;
                    }
                }
            }
        }

        // Not existed module. Add it into profiler list
        if (foundModule == -1)
        {
            mProfiles[nModules].Init(name, mCurrentLevel);
            foundModule = nModules;
            nModules++;
        }
        ASSERT(foundModule >= 0);

        // Mark this module as children of previous profile if neccessary
        if (parentProfile >= 0)
        {
            mProfiles[parentProfile].AddChildren(&mProfiles[foundModule]);
            mProfiles[foundModule].SetParentIdx(parentProfile);
        }

        // Start to record the time
        mCurrentProfile = foundModule;
        ASSERT(mCurrentLevel <= MAX_PROFILE_MODULE);
        mProfiles[foundModule].Start();
    }

    void EndProfile()
    {
        // No profile is active
        if (mCurrentProfile < 0)
        {
            return;
        }

        mProfiles[mCurrentProfile].Stop();

        // Check if this profile has parent
        int parentIdx = mProfiles[mCurrentProfile].getParentIdx();

        // If there is parent return the profile to parent
        if (parentIdx >= 0)
        {
            // Has parent. Return to parent
            mCurrentLevel = mProfiles[parentIdx].GetLevel();
            mCurrentProfile = parentIdx;
        }
        ASSERT(mCurrentLevel >= 0);
    }

    void print(const ProfileTimer* pProfile, CHAR16* pPrintInfo, const unsigned long long& freq)
    {
        if (NULL == pProfile)
        {
            return;
        }
        // First append the top level of profile
        unsigned long long profileTime = pProfile->GetAvgTime() * 1000000 / freq;
        unsigned long long profileCount = pProfile->GetCallCount();
        int level = pProfile->GetLevel();

        CHAR16 moduleName[PROFILE_NAME_WIDTH];
        setMem(moduleName, sizeof(moduleName), 0);
        // Depend on the level set the spacing before name
        for (int i = 0; i < level; i++)
        {
            appendText(moduleName, L"  ");
        }
        appendText(moduleName, pProfile->mName);

        // Padding the naming with specific width
        unsigned int strLen = stringLength(moduleName);
        for (unsigned int i = strLen; i < PROFILE_NAME_WIDTH; i++)
        {
            appendText(moduleName, L" ");
        }

        // Append the module name
        appendText(pPrintInfo, moduleName);
        appendNumber(pPrintInfo, profileTime / 1000, true);
        appendText(pPrintInfo, L".");
        appendNumber(pPrintInfo, profileTime % 1000, false);
        appendText(pPrintInfo, L" ms | ");
        appendNumber(pPrintInfo, profileCount, true);
        appendText(pPrintInfo, L"\r\n");

        // Append the children
        for (int i = 0; i < pProfile->mChildrenCount; i++)
        {
            print(pProfile->mpChildren[i], pPrintInfo, freq);
        }
    }

    void PrintProfile(unsigned long long frequency, const int processorID = -1)
    {
        // Make sure all profile is stopped
        for (int i = 0; i < nModules; i++)
        {
            if (mProfiles[i].isActive())
            {
                mProfiles[i].Stop();
            }
        }

        // Get the profile information
        if (processorID >= 0)
        {
            setText(printLog, L"[Proc ");
            appendNumber(printLog, processorID, false);
            appendText(printLog, L"]");
        }
        else
        {
            setText(printLog, L"[]");
        }

        appendText(printLog, L"Profiler of ");
        appendNumber(printLog, nModules, false);
        appendText(printLog, L" modules. \r\n");
        for (int i = 0; i < nModules; i++)
        {
            // Process the top level. Children will be printed inside
            if (mProfiles[i].mLevel == 0)
            {
                print(&mProfiles[i], printLog, frequency);
            }
        }
        appendText(printLog, L"\r\n");
        logToConsole(printLog);
#if WRITE_PROFILE_MESSAGES_TO_FILE
        unsigned int strSize = stringLength(printLog) * sizeof(CHAR16);
        static bool isNewFile = true;
        if (isNewFile)
        {
            isNewFile = false;
            save(profileFileName, strSize, (unsigned char*)printLog);
        }
        else
        {
            append(profileFileName, strSize, (unsigned char*)printLog);
        }
#endif

    }

    int nModules = 0;
private:
    ProfileTimer mProfiles[MAX_PROFILE_MODULE];
    CHAR16 printLog[2048];
    unsigned int mCurrentLevel = 0;
    int mCurrentProfile = -1;
};

struct ProfilerList
{
    ProfilerList()
    {
        setMem(mActiveProcessor, sizeof(mActiveProcessor), 0);
    }
    void Profile(const CHAR16* name, const unsigned long long processorID)
    {
        mProfiler[processorID].Profile(name);
        mActiveProcessor[processorID] = 1;
    }

    void EndProfile(const unsigned long long processorID)
    {
        mProfiler[processorID].EndProfile();
    }

    void PrintProfile(const unsigned long long frequency)
    {
#if WRITE_PROFILE_MESSAGES_TO_FILE
        {
            timeStampPrefix[0] = (time.Year % 100) / 10 + L'0';
            timeStampPrefix[1] = time.Year % 10 + L'0';
            timeStampPrefix[2] = time.Month / 10 + L'0';
            timeStampPrefix[3] = time.Month % 10 + L'0';
            timeStampPrefix[4] = time.Day / 10 + L'0';
            timeStampPrefix[5] = time.Day % 10 + L'0';
            timeStampPrefix[6] = time.Hour / 10 + L'0';
            timeStampPrefix[7] = time.Hour % 10 + L'0';
            timeStampPrefix[8] = time.Minute / 10 + L'0';
            timeStampPrefix[9] = time.Minute % 10 + L'0';
            timeStampPrefix[10] = time.Second / 10 + L'0';
            timeStampPrefix[11] = time.Second % 10 + L'0';
            timeStampPrefix[13] = 0;
            setText(profileFileName, L"profile_");
            appendText(profileFileName, timeStampPrefix);
            appendText(profileFileName, L".log");

            CHAR16 profileMsg[128];
            setText(profileMsg, L"Write profile into ");
            appendText(profileMsg, profileFileName);
            logToConsole(profileMsg);
        }

#endif

        for (int i = 0; i < MAX_NUMBER_OF_PROCESSORS; i++)
        {
            if (mActiveProcessor[i] && mProfiler[i].nModules > 0)
            {
                mProfiler[i].PrintProfile(frequency, i);
            }
        }
    }

    Profiler mProfiler[MAX_NUMBER_OF_PROCESSORS];
    uint8 mActiveProcessor[MAX_NUMBER_OF_PROCESSORS];
};

static ProfilerList gProfiler;

struct ProfilerRecoder
{
    ProfilerRecoder(const CHAR16* name, const unsigned long long processorID)
    {
        mProcessorID = processorID;
        mEnable = true;
        gProfiler.Profile(name, processorID);
    }

    ProfilerRecoder(const CHAR16* name)
    {
        // Try to get the processor ID. If failed, skip
        mEnable = false;
        EFI_GUID serviceProtocolGuid = EFI_MP_SERVICES_PROTOCOL_GUID;
        EFI_MP_SERVICES_PROTOCOL* pServicesProtocol = NULL;
        bs->LocateProtocol(&serviceProtocolGuid, NULL, (void**)&pServicesProtocol);
        if (EFI_SUCCESS == pServicesProtocol->WhoAmI(pServicesProtocol, &mProcessorID))
        {
            gProfiler.Profile(name, mProcessorID);
            mEnable = true;
        }
    }

    ~ProfilerRecoder()
    {
        if (mEnable)
        {
            gProfiler.EndProfile(mProcessorID);
        }
    }
    unsigned long long mProcessorID;
    bool mEnable = false;
};

#endif

#if ENABLE_PROFILER

// Profile with auto detected processor ID
#define PROFILE_START_PROC(name) ProfilerRecoder recoder(name)

// Profile with known processor ID. This will reduce the overhead of processor ID detection
#define PROFILE_START(name, processorID) ProfilerRecoder recoder(name, processorID)

// Print the profile to file and console
#define PROFILE_PRINT(frequency) { gProfiler.PrintProfile(frequency); }

#else

#define PROFILE_START(name)
#define PROFILE_START(name, processorID)
#define PROFILE_PRINT()
#endif // ENABLE_PROFILER


