using namespace QPI;

constexpr uint64 NOSTROMO_TIER_FACEHUGGER_STAKE_AMOUNT = 20000000ULL;
constexpr uint64 NOSTROMO_TIER_CHESTBURST_STAKE_AMOUNT = 100000000ULL;
constexpr uint64 NOSTROMO_TIER_DOG_STAKE_AMOUNT = 200000000ULL;
constexpr uint64 NOSTROMO_TIER_XENOMORPH_STAKE_AMOUNT = 800000000ULL;
constexpr uint64 NOSTROMO_TIER_WARRIOR_STAKE_AMOUNT = 3200000000ULL;
constexpr uint64 NOSTROMO_QX_TOKEN_ISSUANCE_FEE = 1000000000ULL;

constexpr uint32 NOSTROMO_TIER_FACEHUGGER_POOL_WEIGHT = 55;
constexpr uint32 NOSTROMO_TIER_CHESTBURST_POOL_WEIGHT = 300;
constexpr uint32 NOSTROMO_TIER_DOG_POOL_WEIGHT = 750;
constexpr uint32 NOSTROMO_TIER_XENOMORPH_POOL_WEIGHT = 3050;
constexpr uint32 NOSTROMO_TIER_WARRIOR_POOL_WEIGHT = 13750;

constexpr uint32 NOSTROMO_TIER_FACEHUGGER_UNSTAKE_FEE = 5;
constexpr uint32 NOSTROMO_TIER_CHESTBURST_UNSTAKE_FEE = 4;
constexpr uint32 NOSTROMO_TIER_DOG_UNSTAKE_FEE = 3;
constexpr uint32 NOSTROMO_TIER_XENOMORPH_UNSTAKE_FEE = 2;
constexpr uint32 NOSTROMO_TIER_WARRIOR_UNSTAKE_FEE = 1;
constexpr uint32 NOSTROMO_CREATE_PROJECT_FEE = 100000000;

constexpr uint32 NOSTROMO_MAX_USER = 262144;
constexpr uint32 NOSTROMO_MAX_NUMBER_PROJECT = 262144;
constexpr uint32 NOSTROMO_MAX_NUMBER_TOKEN = 262144;
constexpr uint32 NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST = 128;

struct NOST2
{
};

struct NOST : public ContractBase
{
public:

	/****** PORTED TIMEUTILS FROM OLD QUOTTERY *****/
	/**
	 * Compare 2 date in uint32 format
	 * @return -1 lesser(ealier) A<B, 0 equal A=B, 1 greater(later) A>B
	 */
	inline static sint32 dateCompare(uint32& A, uint32& B, sint32& i)
	{
		if (A == B) return 0;
		if (A < B) return -1;
		return 1;
	}

	/**
	 * @return pack qtry datetime data from year, month, day, hour, minute, second to a uint32
	 * year is counted from 24 (2024)
	 */
	inline static void packQuotteryDate(uint32 _year, uint32 _month, uint32 _day, uint32 _hour, uint32 _minute, uint32 _second, uint32& res)
	{
		res = ((_year - 24) << 26) | (_month << 22) | (_day << 17) | (_hour << 12) | (_minute << 6) | (_second);
	}

	inline static uint32 qtryGetYear(uint32 data)
	{
		return ((data >> 26) + 24);
	}
	inline static uint32 qtryGetMonth(uint32 data)
	{
		return ((data >> 22) & 0b1111);
	}
	inline static uint32 qtryGetDay(uint32 data)
	{
		return ((data >> 17) & 0b11111);
	}
	inline static uint32 qtryGetHour(uint32 data)
	{
		return ((data >> 12) & 0b11111);
	}
	inline static uint32 qtryGetMinute(uint32 data)
	{
		return ((data >> 6) & 0b111111);
	}
	inline static uint32 qtryGetSecond(uint32 data)
	{
		return (data & 0b111111);
	}
	/*
	* @return unpack qtry datetime from uin32 to year, month, day, hour, minute, secon
	*/
	inline static void unpackQuotteryDate(uint8& _year, uint8& _month, uint8& _day, uint8& _hour, uint8& _minute, uint8& _second, uint32 data)
	{
		_year = qtryGetYear(data); // 6 bits
		_month = qtryGetMonth(data); //4bits
		_day = qtryGetDay(data); //5bits
		_hour = qtryGetHour(data); //5bits
		_minute = qtryGetMinute(data); //6bits
		_second = qtryGetSecond(data); //6bits
	}

	inline static void accumulatedDay(sint32 month, uint64& res)
	{
		switch (month)
		{
		case 1: res = 0; break;
		case 2: res = 31; break;
		case 3: res = 59; break;
		case 4: res = 90; break;
		case 5: res = 120; break;
		case 6: res = 151; break;
		case 7: res = 181; break;
		case 8: res = 212; break;
		case 9: res = 243; break;
		case 10:res = 273; break;
		case 11:res = 304; break;
		case 12:res = 334; break;
		}
	}
	/**
	 * @return difference in number of second, A must be smaller than or equal B to have valid value
	 */
	inline static void diffDateInSecond(uint32& A, uint32& B, sint32& i, uint64& dayA, uint64& dayB, uint64& res)
	{
		if (dateCompare(A, B, i) >= 0)
		{
			res = 0;
			return;
		}
		accumulatedDay(qtryGetMonth(A), dayA);
		dayA += qtryGetDay(A);
		accumulatedDay(qtryGetMonth(B), dayB);
		dayB += (qtryGetYear(B) - qtryGetYear(A)) * 365ULL + qtryGetDay(B);

		// handling leap-year: only store last 2 digits of year here, don't care about mod 100 & mod 400 case
		for (i = qtryGetYear(A); (uint32)(i) < qtryGetYear(B); i++)
		{
			if (mod(i, 4) == 0)
			{
				dayB++;
			}
		}
		if (mod(sint32(qtryGetYear(A)), 4) == 0 && (qtryGetMonth(A) > 2)) dayA++;
		if (mod(sint32(qtryGetYear(B)), 4) == 0 && (qtryGetMonth(B) > 2)) dayB++;
		res = (dayB - dayA) * 3600ULL * 24;
		res += (qtryGetHour(B) * 3600 + qtryGetMinute(B) * 60 + qtryGetSecond(B));
		res -= (qtryGetHour(A) * 3600 + qtryGetMinute(A) * 60 + qtryGetSecond(A));
	}

	inline static bool checkValidQtryDateTime(uint32& A)
	{
		if (qtryGetMonth(A) > 12) return false;
		if (qtryGetDay(A) > 31) return false;
		if ((qtryGetDay(A) == 31) &&
			(qtryGetMonth(A) != 1) && (qtryGetMonth(A) != 3) && (qtryGetMonth(A) != 5) &&
			(qtryGetMonth(A) != 7) && (qtryGetMonth(A) != 8) && (qtryGetMonth(A) != 10) && (qtryGetMonth(A) != 12)) return false;
		if ((qtryGetDay(A) == 30) && (qtryGetMonth(A) == 2)) return false;
		if ((qtryGetDay(A) == 29) && (qtryGetMonth(A) == 2) && (mod(qtryGetYear(A), 4u) != 0)) return false;
		if (qtryGetHour(A) >= 24) return false;
		if (qtryGetMinute(A) >= 60) return false;
		if (qtryGetSecond(A) >= 60) return false;
		return true;
	}

	/**
	* @return Current date from core node system
	*/

	inline static void getCurrentDate(const QPI::QpiContextFunctionCall& qpi, uint32& res)
	{
		packQuotteryDate(qpi.year(), qpi.month(), qpi.day(), qpi.hour(), qpi.minute(), qpi.second(), res);
	}
	/****** END PORTED TIMEUTILS FROM OLD QUOTTERY *****/

	struct registerInTier_input
	{
		uint32 tierLevel;
	};

	struct registerInTier_output
	{
		uint32 tierLevel;
	};

	struct logoutFromTier_input
	{

	};

	struct logoutFromTier_output
	{
		bit result;
	};

	struct createProject_input
	{
		uint64 tokenName;
		uint64 supply;
		uint32 startYear;
		uint32 startMonth;
		uint32 startDay;
		uint32 startHour;
		uint32 endYear;
		uint32 endMonth;
		uint32 endDay;
		uint32 endHour;
	};

	struct createProject_output
	{
		uint32 indexOfProject;
	};

	struct voteInProject_input
	{
		uint32 indexOfProject;
		bit decision;
	};

	struct voteInProject_output
	{

	};

	struct createFundraising_input
	{
		uint64 tokenPrice;
		uint64 soldAmount;
		uint64 requiredFunds;

		uint32 indexOfProject;
		uint32 firstPhaseStartYear;
		uint32 firstPhaseStartMonth;
		uint32 firstPhaseStartDay;
		uint32 firstPhaseStartHour;
		uint32 firstPhaseEndYear;
		uint32 firstPhaseEndMonth;
		uint32 firstPhaseEndDay;
		uint32 firstPhaseEndHour;

		uint32 secondPhaseStartYear;
		uint32 secondPhaseStartMonth;
		uint32 secondPhaseStartDay;
		uint32 secondPhaseStartHour;
		uint32 secondPhaseEndYear;
		uint32 secondPhaseEndMonth;
		uint32 secondPhaseEndDay;
		uint32 secondPhaseEndHour;

		uint32 thirdPhaseStartYear;
		uint32 thirdPhaseStartMonth;
		uint32 thirdPhaseStartDay;
		uint32 thirdPhaseStartHour;
		uint32 thirdPhaseEndYear;
		uint32 thirdPhaseEndMonth;
		uint32 thirdPhaseEndDay;
		uint32 thirdPhaseEndHour;

		uint32 listingStartYear;
		uint32 listingStartMonth;
		uint32 listingStartDay;
		uint32 listingStartHour;

		uint32 cliffEndYear;
		uint32 cliffEndMonth;
		uint32 cliffEndDay;
		uint32 cliffEndHour;

		uint32 vestingEndYear;
		uint32 vestingEndMonth;
		uint32 vestingEndDay;
		uint32 vestingEndHour;

		uint8 threshold;
		uint8 TGE;
		uint8 stepOfVesting;
	};

	struct createFundraising_output
	{

	};

	struct investInProject_input
	{
		uint32 indexOfFundraising;
	};

	struct investInProject_output
	{

	};

	struct claimToken_input
	{
		uint64 amount;
		uint32 indexOfFundraising;
	};

	struct claimToken_output
	{
		uint64 claimedAmount;
	};

	struct upgradeTier_input
	{
		uint32 newTierLevel;
	};

	struct upgradeTier_output
	{

	};

	struct TransferShareManagementRights_input
	{
		Asset asset;
		sint64 numberOfShares;
		uint32 newManagingContractIndex;
	};
	struct TransferShareManagementRights_output
	{
		sint64 transferredNumberOfShares;
	};

	struct getStats_input
	{

	};

	struct getStats_output
	{
		uint64 epochRevenue, totalPoolWeight;
		uint32 numberOfRegister, numberOfCreatedProject, numberOfFundraising;
	};

	struct getTierLevelByUser_input
	{
		id userId;
	};

	struct getTierLevelByUser_output
	{
		uint8 tierLevel;
	};

	struct getUserVoteStatus_input
	{
		id userId;
	};

	struct getUserVoteStatus_output
	{
		uint32 numberOfVotedProjects;
		Array<uint32, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST> projectIndexList;
	};

	struct checkTokenCreatability_input
	{
		uint64 tokenName;
	};

	struct checkTokenCreatability_output
	{
		bit result;             // result = 1 is the token already issued by SC
	};

	struct getNumberOfInvestedProjects_input
	{
		id userId;
	};

	struct getNumberOfInvestedProjects_output
	{
		uint32 numberOfInvestedProjects;
	};

protected:
	HashMap<id, uint8, NOSTROMO_MAX_USER> users;
	HashMap<id, Array<uint32, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST>, NOSTROMO_MAX_USER> voteStatus;
	HashMap<id, uint32, NOSTROMO_MAX_USER> numberOfVotedProject;
	HashSet<uint64, NOSTROMO_MAX_NUMBER_TOKEN> tokens;

	struct investInfo
	{
		uint64 investedAmount;
		uint64 claimedAmount;
		uint32 indexOfFundraising;
	}; 

	HashMap<id, Array<investInfo, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST>, NOSTROMO_MAX_USER> investors;
	HashMap<id, uint32, NOSTROMO_MAX_USER> numberOfInvestedProjects;
	Array<investInfo, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST> tmpInvestedList;

	struct projectInfo
	{
		id creator;
		uint64 tokenName;
		uint64 supplyOfToken;
		uint32 startDate;
		uint32 endDate;
		uint32 numberOfYes;
		uint32 numberOfNo;
		bit isCreatedFundarasing;
	};

	Array<projectInfo, NOSTROMO_MAX_NUMBER_PROJECT> projects;

	struct fundaraisingInfo
	{
		uint64 tokenPrice;
		uint64 soldAmount;
		uint64 requiredFunds;
		uint64 raisedFunds;
		uint32 indexOfProject;
		uint32 firstPhaseStartDate;
		uint32 firstPhaseEndDate;
		uint32 secondPhaseStartDate;
		uint32 secondPhaseEndDate;
		uint32 thirdPhaseStartDate;
		uint32 thirdPhaseEndDate;
		uint32 listingStartDate;
		uint32 cliffEndDate;
		uint32 vestingEndDate;
		uint8 threshold;
		uint8 TGE;
		uint8 stepOfVesting;
		bit isCreatedToken;
	};

	Array<fundaraisingInfo, NOSTROMO_MAX_NUMBER_PROJECT> fundaraisings;

	id teamAddress;
	sint64 transferRightsFee;
	uint64 epochRevenue, totalPoolWeight;
	uint32 numberOfRegister, numberOfCreatedProject, numberOfFundraising;

	struct registerInTier_locals
	{
		uint64 tierStakedAmount;
		uint32 poolWeight;
	};

	PUBLIC_PROCEDURE_WITH_LOCALS(registerInTier)
	{
		if (state.users.contains(qpi.invocator()))
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		if (input.tierLevel < 1 || input.tierLevel > 5)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		
		switch (input.tierLevel)
		{
		case 1:
			locals.tierStakedAmount = NOSTROMO_TIER_FACEHUGGER_STAKE_AMOUNT;
			locals.poolWeight = NOSTROMO_TIER_FACEHUGGER_POOL_WEIGHT;
			break;
		case 2:
			locals.tierStakedAmount = NOSTROMO_TIER_CHESTBURST_STAKE_AMOUNT;
			locals.poolWeight = NOSTROMO_TIER_CHESTBURST_POOL_WEIGHT;
			break;
		case 3:
			locals.tierStakedAmount = NOSTROMO_TIER_DOG_STAKE_AMOUNT;
			locals.poolWeight = NOSTROMO_TIER_DOG_POOL_WEIGHT;
			break;
		case 4:
			locals.tierStakedAmount = NOSTROMO_TIER_XENOMORPH_STAKE_AMOUNT;
			locals.poolWeight = NOSTROMO_TIER_XENOMORPH_POOL_WEIGHT;
			break;
		case 5:
			locals.tierStakedAmount = NOSTROMO_TIER_WARRIOR_STAKE_AMOUNT;
			locals.poolWeight = NOSTROMO_TIER_WARRIOR_POOL_WEIGHT;
			break;
		default:
			break;
		}
		if (qpi.invocationReward() < (sint64)locals.tierStakedAmount)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		else 
		{
			state.users.set(qpi.invocator(), input.tierLevel);
			state.numberOfRegister++;
			if (qpi.invocationReward() > (sint64)locals.tierStakedAmount)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward() - locals.tierStakedAmount);
			}
			state.totalPoolWeight += locals.poolWeight;
			output.tierLevel = input.tierLevel;
		}
	}

	struct logoutFromTier_locals
	{
		uint64 earnedAmount;
		uint32 elementIndex;
		uint8 tierLevel;
	};

	PUBLIC_PROCEDURE_WITH_LOCALS(logoutFromTier)
	{
		if (state.users.contains(qpi.invocator()) == 0)
		{
			return ;
		}
		state.users.get(qpi.invocator(), locals.tierLevel);
		switch (locals.tierLevel)
		{
		case 1:
			locals.earnedAmount = div(NOSTROMO_TIER_FACEHUGGER_STAKE_AMOUNT * NOSTROMO_TIER_FACEHUGGER_UNSTAKE_FEE, 100ULL);
			qpi.transfer(qpi.invocator(), qpi.invocationReward() + NOSTROMO_TIER_FACEHUGGER_STAKE_AMOUNT - locals.earnedAmount);
			state.epochRevenue += locals.earnedAmount;
			state.totalPoolWeight -= NOSTROMO_TIER_FACEHUGGER_POOL_WEIGHT;
			break;
		case 2:
			locals.earnedAmount = div(NOSTROMO_TIER_CHESTBURST_STAKE_AMOUNT * NOSTROMO_TIER_CHESTBURST_UNSTAKE_FEE, 100ULL);
			qpi.transfer(qpi.invocator(), qpi.invocationReward() + NOSTROMO_TIER_CHESTBURST_STAKE_AMOUNT - locals.earnedAmount);
			state.epochRevenue += locals.earnedAmount;
			state.totalPoolWeight -= NOSTROMO_TIER_CHESTBURST_POOL_WEIGHT;
			break;
		case 3:
			locals.earnedAmount = div(NOSTROMO_TIER_DOG_STAKE_AMOUNT * NOSTROMO_TIER_DOG_UNSTAKE_FEE, 100ULL);
			qpi.transfer(qpi.invocator(), qpi.invocationReward() + NOSTROMO_TIER_DOG_STAKE_AMOUNT - locals.earnedAmount);
			state.epochRevenue += locals.earnedAmount;
			state.totalPoolWeight -= NOSTROMO_TIER_DOG_POOL_WEIGHT;
			break;
		case 4:
			locals.earnedAmount = div(NOSTROMO_TIER_XENOMORPH_STAKE_AMOUNT * NOSTROMO_TIER_XENOMORPH_UNSTAKE_FEE, 100ULL);
			qpi.transfer(qpi.invocator(), qpi.invocationReward() + NOSTROMO_TIER_XENOMORPH_STAKE_AMOUNT - locals.earnedAmount);
			state.epochRevenue += locals.earnedAmount;
			state.totalPoolWeight -= NOSTROMO_TIER_XENOMORPH_POOL_WEIGHT;
			break;
		case 5:
			locals.earnedAmount = div(NOSTROMO_TIER_WARRIOR_STAKE_AMOUNT * NOSTROMO_TIER_WARRIOR_UNSTAKE_FEE, 100ULL);
			qpi.transfer(qpi.invocator(), qpi.invocationReward() + NOSTROMO_TIER_WARRIOR_STAKE_AMOUNT - locals.earnedAmount);
			state.epochRevenue += locals.earnedAmount;
			state.totalPoolWeight -= NOSTROMO_TIER_WARRIOR_POOL_WEIGHT;
			break;
		default:
			break;
		}

		state.users.removeByKey(qpi.invocator());
		state.numberOfRegister -= 1;
		output.result = 1;
	}

	struct createProject_locals
	{
		projectInfo newProject;
		uint32 elementIndex, startDate, endDate, curDate;
		uint8 tierLevel;
	};

	PUBLIC_PROCEDURE_WITH_LOCALS(createProject)
	{
		packQuotteryDate(input.startYear, input.startMonth, input.startDay, input.startHour, 0, 0, locals.startDate);
		packQuotteryDate(input.endYear, input.endMonth, input.endDay, input.endHour, 0, 0, locals.endDate);
		packQuotteryDate(qpi.year(), qpi.month(), qpi.day(), qpi.hour(), qpi.minute(), qpi.second(), locals.curDate);

		if(locals.curDate > locals.startDate || locals.startDate >= locals.endDate || checkValidQtryDateTime(locals.startDate) == 0 || checkValidQtryDateTime(locals.endDate) == 0)
		{
			output.indexOfProject = NOSTROMO_MAX_NUMBER_PROJECT;
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return;
		}

		if (state.tokens.contains(input.tokenName))
		{
			output.indexOfProject = NOSTROMO_MAX_NUMBER_PROJECT;
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}

		if (state.users.get(qpi.invocator(), locals.tierLevel) && (locals.tierLevel == 4 || locals.tierLevel == 5))
		{
			if (qpi.invocationReward() < NOSTROMO_CREATE_PROJECT_FEE)
			{
				if (qpi.invocationReward() > 0)
				{
					qpi.transfer(qpi.invocator(), qpi.invocationReward());
				}
				output.indexOfProject = NOSTROMO_MAX_NUMBER_PROJECT;
				return ;
			}
			if (qpi.invocationReward() > NOSTROMO_CREATE_PROJECT_FEE)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward() - NOSTROMO_CREATE_PROJECT_FEE);
			}
			state.epochRevenue += NOSTROMO_CREATE_PROJECT_FEE;
			
			locals.newProject.creator = qpi.invocator();
			locals.newProject.tokenName = input.tokenName;
			locals.newProject.supplyOfToken = input.supply;
			locals.newProject.startDate = locals.startDate;
			locals.newProject.endDate = locals.endDate;
			locals.newProject.numberOfYes = 0;
			locals.newProject.numberOfNo = 0;

			output.indexOfProject = state.numberOfCreatedProject;
			state.projects.set(state.numberOfCreatedProject++, locals.newProject);
			state.tokens.add(input.tokenName);
		}
		else 
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			output.indexOfProject = NOSTROMO_MAX_NUMBER_PROJECT;
		}
	}

	struct voteInProject_locals
	{
		projectInfo votedProject;
		Array<uint32, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST> votedList;
		uint32 elementIndex, curDate, numberOfVotedProject, i;
		bit flag;
	};

	PUBLIC_PROCEDURE_WITH_LOCALS(voteInProject)
	{
		if (input.indexOfProject >= state.numberOfCreatedProject)
		{
			return ;
		}
		if (state.users.contains(qpi.invocator()) == 0)
		{
			return ;
		}
		state.numberOfVotedProject.get(qpi.invocator(), locals.numberOfVotedProject);
		if (locals.numberOfVotedProject == NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST)
		{
			return ;
		}
		state.voteStatus.get(qpi.invocator(), locals.votedList);
		for (locals.i = 0; locals.i < locals.numberOfVotedProject; locals.i++)
		{
			if (locals.votedList.get(locals.i) == input.indexOfProject)
			{
				return ;
			}
		}
		packQuotteryDate(qpi.year(), qpi.month(), qpi.day(), qpi.hour(), qpi.minute(), qpi.second(), locals.curDate);
		if (locals.curDate >= state.projects.get(input.indexOfProject).startDate && locals.curDate < state.projects.get(input.indexOfProject).endDate)
		{
			locals.votedProject = state.projects.get(input.indexOfProject);
			if (input.decision)
			{
				locals.votedProject.numberOfYes++;
			}
			else 
			{
				locals.votedProject.numberOfNo++;
			}
			state.projects.set(input.indexOfProject, locals.votedProject);
			locals.votedList.set(locals.numberOfVotedProject++, input.indexOfProject);
			state.voteStatus.set(qpi.invocator(), locals.votedList);
			state.numberOfVotedProject.set(qpi.invocator(), locals.numberOfVotedProject);
		}
	}

	struct createFundraising_locals
	{
		projectInfo tmpProject;
		fundaraisingInfo newFundraising;
		uint32 curDate, firstPhaseStartDate, firstPhaseEndDate, secondPhaseStartDate, secondPhaseEndDate, thirdPhaseStartDate, thirdPhaseEndDate, listingStartDate, cliffEndDate, vestingEndDate;
	};

	PUBLIC_PROCEDURE_WITH_LOCALS(createFundraising)
	{
		packQuotteryDate(qpi.year(), qpi.month(), qpi.day(), qpi.hour(), qpi.minute(), qpi.second(), locals.curDate);
		packQuotteryDate(input.firstPhaseStartYear, input.firstPhaseStartMonth, input.firstPhaseStartDay, input.firstPhaseStartHour, 0, 0, locals.firstPhaseStartDate);
		packQuotteryDate(input.secondPhaseStartYear, input.secondPhaseStartMonth, input.secondPhaseStartDay, input.secondPhaseStartHour, 0, 0, locals.secondPhaseStartDate);
		packQuotteryDate(input.thirdPhaseStartYear, input.thirdPhaseStartMonth, input.thirdPhaseStartDay, input.thirdPhaseStartHour, 0, 0, locals.thirdPhaseStartDate);
		packQuotteryDate(input.firstPhaseEndYear, input.firstPhaseEndMonth, input.firstPhaseEndDay, input.firstPhaseEndHour, 0, 0, locals.firstPhaseEndDate);
		packQuotteryDate(input.secondPhaseEndYear, input.secondPhaseEndMonth, input.secondPhaseEndDay, input.secondPhaseEndHour, 0, 0, locals.secondPhaseEndDate);
		packQuotteryDate(input.thirdPhaseEndYear, input.thirdPhaseEndMonth, input.thirdPhaseEndDay, input.thirdPhaseEndHour, 0, 0, locals.thirdPhaseEndDate);
		packQuotteryDate(input.listingStartYear, input.listingStartMonth, input.listingStartDay, input.listingStartHour, 0, 0, locals.listingStartDate);
		packQuotteryDate(input.cliffEndYear, input.cliffEndMonth, input.cliffEndDay, input.cliffEndHour, 0, 0, locals.cliffEndDate);
		packQuotteryDate(input.vestingEndYear, input.vestingEndMonth, input.vestingEndDay, input.vestingEndHour, 0, 0, locals.vestingEndDate);

		if (locals.curDate > locals.firstPhaseStartDate || locals.firstPhaseStartDate >= locals.firstPhaseEndDate || locals.firstPhaseEndDate > locals.secondPhaseStartDate || locals.secondPhaseStartDate >= locals.secondPhaseEndDate || locals.secondPhaseEndDate > locals.thirdPhaseStartDate || locals.thirdPhaseStartDate >= locals.thirdPhaseEndDate || locals.thirdPhaseEndDate > locals.listingStartDate || locals.listingStartDate > locals.cliffEndDate || locals.cliffEndDate > locals.vestingEndDate)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		if (checkValidQtryDateTime(locals.firstPhaseStartDate) == 0 || checkValidQtryDateTime(locals.firstPhaseEndDate) == 0 || checkValidQtryDateTime(locals.secondPhaseStartDate) == 0 || checkValidQtryDateTime(locals.secondPhaseEndDate) == 0 || checkValidQtryDateTime(locals.thirdPhaseStartDate) == 0 || checkValidQtryDateTime(locals.thirdPhaseEndDate) == 0 || checkValidQtryDateTime(locals.listingStartDate) == 0 || checkValidQtryDateTime(locals.cliffEndDate) == 0 || checkValidQtryDateTime(locals.vestingEndDate) == 0)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}

		if (input.stepOfVesting == 0 || input.stepOfVesting > 12 || input.TGE > 50 || input.threshold > 50 || input.indexOfProject >= state.numberOfCreatedProject)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		

		if (state.projects.get(input.indexOfProject).creator != qpi.invocator())
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}

		if (input.soldAmount > state.projects.get(input.indexOfProject).supplyOfToken)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}

		if (locals.curDate <= state.projects.get(input.indexOfProject).endDate || state.projects.get(input.indexOfProject).numberOfYes <= state.projects.get(input.indexOfProject).numberOfNo || state.projects.get(input.indexOfProject).isCreatedFundarasing == 1)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}

		if (input.tokenPrice * input.soldAmount < input.requiredFunds + div(input.requiredFunds * input.threshold, 100ULL))
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		
		if (qpi.invocationReward() < NOSTROMO_QX_TOKEN_ISSUANCE_FEE)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}

		if (qpi.invocationReward() > NOSTROMO_QX_TOKEN_ISSUANCE_FEE)
		{
			qpi.transfer(qpi.invocator(), NOSTROMO_QX_TOKEN_ISSUANCE_FEE - qpi.invocationReward());
		}
		
		locals.tmpProject = state.projects.get(input.indexOfProject);
		locals.tmpProject.isCreatedFundarasing = 1;
		state.projects.set(input.indexOfProject, locals.tmpProject);
		
		locals.newFundraising.tokenPrice = input.tokenPrice;
		locals.newFundraising.soldAmount = input.soldAmount;
		locals.newFundraising.requiredFunds = input.requiredFunds;
		locals.newFundraising.raisedFunds = 0;
		locals.newFundraising.indexOfProject = input.indexOfProject;
		locals.newFundraising.firstPhaseStartDate = locals.firstPhaseStartDate;
		locals.newFundraising.firstPhaseEndDate = locals.firstPhaseEndDate;
		locals.newFundraising.secondPhaseStartDate = locals.secondPhaseStartDate;
		locals.newFundraising.secondPhaseEndDate = locals.secondPhaseEndDate;
		locals.newFundraising.thirdPhaseStartDate = locals.thirdPhaseStartDate;
		locals.newFundraising.thirdPhaseEndDate = locals.thirdPhaseEndDate;
		locals.newFundraising.listingStartDate = locals.listingStartDate;
		locals.newFundraising.cliffEndDate = locals.cliffEndDate;
		locals.newFundraising.vestingEndDate = locals.vestingEndDate;
		locals.newFundraising.threshold = input.threshold;
		locals.newFundraising.TGE = input.TGE;
		locals.newFundraising.stepOfVesting = input.stepOfVesting;

		state.fundaraisings.set(state.numberOfFundraising++, locals.newFundraising);
	}

	struct investInProject_locals
	{
		QX::IssueAsset_input input;
		QX::IssueAsset_output output;
		QX::TransferShareManagementRights_input TransferShareManagementRightsInput;
		QX::TransferShareManagementRights_output TransferShareManagementRightsOutput;
		investInfo tmpInvestData;
		fundaraisingInfo tmpFundraising;
		uint64 maxCap, minCap, maxInvestmentPerUser, userInvestedAmount;
		uint32 curDate, elementIndex, i, numberOfInvestedProjects;
		uint8 tierLevel;
	};

	PUBLIC_PROCEDURE_WITH_LOCALS(investInProject)
	{
		if (input.indexOfFundraising >= state.numberOfFundraising || qpi.invocationReward() == 0)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}

		locals.maxCap = state.fundaraisings.get(input.indexOfFundraising).requiredFunds + div(state.fundaraisings.get(input.indexOfFundraising).requiredFunds * state.fundaraisings.get(input.indexOfFundraising).threshold, 100ULL);
		locals.minCap = state.fundaraisings.get(input.indexOfFundraising).requiredFunds - div(state.fundaraisings.get(input.indexOfFundraising).requiredFunds * state.fundaraisings.get(input.indexOfFundraising).threshold, 100ULL);
		if (state.fundaraisings.get(input.indexOfFundraising).raisedFunds + qpi.invocationReward() > locals.maxCap)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		if (state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects) && locals.numberOfInvestedProjects >= NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		
		packQuotteryDate(qpi.year(), qpi.month(), qpi.day(), qpi.hour(), qpi.minute(), qpi.second(), locals.curDate);

		locals.tmpFundraising = state.fundaraisings.get(input.indexOfFundraising);

		if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).firstPhaseStartDate && locals.curDate < state.fundaraisings.get(input.indexOfFundraising).firstPhaseEndDate)
		{
			if (state.users.contains(qpi.invocator()) == 0)
			{
				if (qpi.invocationReward() > 0)
				{	
					qpi.transfer(qpi.invocator(), qpi.invocationReward());
				}
				return ;
			}
			
			state.users.get(qpi.invocator(), locals.tierLevel);
			switch (locals.tierLevel)
			{
			case 1:
				locals.maxInvestmentPerUser = div(locals.maxCap * NOSTROMO_TIER_FACEHUGGER_POOL_WEIGHT, state.totalPoolWeight);
				break;
			case 2:
				locals.maxInvestmentPerUser = div(locals.maxCap * NOSTROMO_TIER_CHESTBURST_POOL_WEIGHT, state.totalPoolWeight);
				break;
			case 3:
				locals.maxInvestmentPerUser = div(locals.maxCap * NOSTROMO_TIER_DOG_POOL_WEIGHT, state.totalPoolWeight);
				break;
			case 4:
				locals.maxInvestmentPerUser = div(locals.maxCap * NOSTROMO_TIER_XENOMORPH_POOL_WEIGHT, state.totalPoolWeight);
				break;
			case 5:
				locals.maxInvestmentPerUser = div(locals.maxCap * NOSTROMO_TIER_WARRIOR_POOL_WEIGHT, state.totalPoolWeight);
				break;
			default:
				break;
			}

			state.investors.get(qpi.invocator(), state.tmpInvestedList);
			state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects);

			for (locals.i = 0; locals.i < locals.numberOfInvestedProjects; locals.i++)
			{
				if (state.tmpInvestedList.get(locals.i).indexOfFundraising == input.indexOfFundraising)
				{
					locals.userInvestedAmount = state.tmpInvestedList.get(locals.i).investedAmount;
					break;
				}
			}
			
			locals.tmpInvestData.indexOfFundraising = input.indexOfFundraising;

			if (locals.i < locals.numberOfInvestedProjects)
			{
				if (qpi.invocationReward() + locals.userInvestedAmount > locals.maxInvestmentPerUser)
				{
					qpi.transfer(qpi.invocator(), qpi.invocationReward() + locals.userInvestedAmount - locals.maxInvestmentPerUser);
					
					locals.tmpInvestData.investedAmount = locals.maxInvestmentPerUser;
					locals.tmpFundraising.raisedFunds += locals.maxInvestmentPerUser - locals.userInvestedAmount;
				}
				else 
				{
					locals.tmpInvestData.investedAmount = qpi.invocationReward() + locals.userInvestedAmount;
					locals.tmpFundraising.raisedFunds += qpi.invocationReward();
				}
				state.tmpInvestedList.set(locals.i, locals.tmpInvestData);
				state.investors.set(qpi.invocator(), state.tmpInvestedList);
			}
			else 
			{
				if (qpi.invocationReward() > (sint64)locals.maxInvestmentPerUser)
				{
					qpi.transfer(qpi.invocator(), qpi.invocationReward() - locals.maxInvestmentPerUser);
					locals.tmpInvestData.investedAmount = locals.maxInvestmentPerUser;				
					locals.tmpFundraising.raisedFunds += locals.maxInvestmentPerUser;
				}
				else 
				{
					locals.tmpInvestData.investedAmount = qpi.invocationReward();		
					locals.tmpFundraising.raisedFunds += qpi.invocationReward();
				}

				state.tmpInvestedList.set(locals.numberOfInvestedProjects, locals.tmpInvestData);
				state.investors.set(qpi.invocator(), state.tmpInvestedList);
				if (state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects))
				{
					state.numberOfInvestedProjects.set(qpi.invocator(), locals.numberOfInvestedProjects + 1);
				}
				else 
				{
					state.numberOfInvestedProjects.set(qpi.invocator(), 1);
				}
			}
		}
		else if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).secondPhaseStartDate && locals.curDate < state.fundaraisings.get(input.indexOfFundraising).secondPhaseEndDate)
		{
			if (state.users.contains(qpi.invocator()) == 0)
			{
				if (qpi.invocationReward() > 0)
				{	
					qpi.transfer(qpi.invocator(), qpi.invocationReward());
				}
				return ;
			}
			
			state.users.get(qpi.invocator(), locals.tierLevel);
			if (locals.tierLevel < 4)
			{
				if (qpi.invocationReward() > 0)
				{
					qpi.transfer(qpi.invocator(), qpi.invocationReward());
				}
				return ;
			}
			switch (locals.tierLevel)
			{
			case 4:
				locals.maxInvestmentPerUser = div(locals.maxCap * NOSTROMO_TIER_XENOMORPH_POOL_WEIGHT, state.totalPoolWeight);
				break;
			case 5:
				locals.maxInvestmentPerUser = div(locals.maxCap * NOSTROMO_TIER_WARRIOR_POOL_WEIGHT, state.totalPoolWeight);
				break;
			default:
				break;
			}

			state.investors.get(qpi.invocator(), state.tmpInvestedList);
			state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects);

			for (locals.i = 0; locals.i < locals.numberOfInvestedProjects; locals.i++)
			{
				if (state.tmpInvestedList.get(locals.i).indexOfFundraising == input.indexOfFundraising)
				{
					locals.userInvestedAmount = state.tmpInvestedList.get(locals.i).investedAmount;
					break;
				}
			}
			
			locals.tmpInvestData.indexOfFundraising = input.indexOfFundraising;

			if (locals.i < locals.numberOfInvestedProjects)
			{
				if (qpi.invocationReward() + locals.userInvestedAmount > locals.maxInvestmentPerUser)
				{
					qpi.transfer(qpi.invocator(), qpi.invocationReward() + locals.userInvestedAmount - locals.maxInvestmentPerUser);
					
					locals.tmpInvestData.investedAmount = locals.maxInvestmentPerUser;
					locals.tmpFundraising.raisedFunds += locals.maxInvestmentPerUser - locals.userInvestedAmount;
				}
				else 
				{
					locals.tmpInvestData.investedAmount = qpi.invocationReward() + locals.userInvestedAmount;
					locals.tmpFundraising.raisedFunds += qpi.invocationReward();
				}
				state.tmpInvestedList.set(locals.i, locals.tmpInvestData);
				state.investors.set(qpi.invocator(), state.tmpInvestedList);
			}
			else 
			{
				if (qpi.invocationReward() > (sint64)locals.maxInvestmentPerUser)
				{
					qpi.transfer(qpi.invocator(), qpi.invocationReward() - locals.maxInvestmentPerUser);
					locals.tmpInvestData.investedAmount = locals.maxInvestmentPerUser;				
					locals.tmpFundraising.raisedFunds += locals.maxInvestmentPerUser;
				}
				else 
				{
					locals.tmpInvestData.investedAmount = qpi.invocationReward();		
					locals.tmpFundraising.raisedFunds += qpi.invocationReward();
				}

				state.tmpInvestedList.set(locals.numberOfInvestedProjects, locals.tmpInvestData);
				state.investors.set(qpi.invocator(), state.tmpInvestedList);
				if (state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects))
				{
					state.numberOfInvestedProjects.set(qpi.invocator(), locals.numberOfInvestedProjects + 1);
				}
				else 
				{
					state.numberOfInvestedProjects.set(qpi.invocator(), 1);
				}
			}
		}
		else if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).thirdPhaseStartDate && locals.curDate < state.fundaraisings.get(input.indexOfFundraising).thirdPhaseEndDate)
		{
			state.investors.get(qpi.invocator(), state.tmpInvestedList);
			state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects);

			for (locals.i = 0; locals.i < locals.numberOfInvestedProjects; locals.i++)
			{
				if (state.tmpInvestedList.get(locals.i).indexOfFundraising == input.indexOfFundraising)
				{
					locals.userInvestedAmount = state.tmpInvestedList.get(locals.i).investedAmount;
					break;
				}
			}

			locals.tmpInvestData.indexOfFundraising = input.indexOfFundraising;

			if (locals.i < locals.numberOfInvestedProjects)
			{
				locals.tmpInvestData.investedAmount = qpi.invocationReward() + locals.userInvestedAmount;
				state.tmpInvestedList.set(locals.i, locals.tmpInvestData);
				state.investors.set(qpi.invocator(), state.tmpInvestedList);
			}
			else 
			{
				locals.tmpInvestData.investedAmount = qpi.invocationReward();

				state.tmpInvestedList.set(locals.numberOfInvestedProjects, locals.tmpInvestData);
				state.investors.set(qpi.invocator(), state.tmpInvestedList);
				if (state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects))
				{
					state.numberOfInvestedProjects.set(qpi.invocator(), locals.numberOfInvestedProjects + 1);
				}
				else 
				{
					state.numberOfInvestedProjects.set(qpi.invocator(), 1);
				}
			}
			locals.tmpFundraising.raisedFunds += qpi.invocationReward();
		}
		else
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		if (locals.minCap <= locals.tmpFundraising.raisedFunds && locals.tmpFundraising.isCreatedToken == 0)
		{
			locals.input.assetName = state.projects.get(locals.tmpFundraising.indexOfProject).tokenName;
			locals.input.numberOfDecimalPlaces = 0;
			locals.input.numberOfShares = state.projects.get(locals.tmpFundraising.indexOfProject).supplyOfToken;
			locals.input.unitOfMeasurement = 0;

			INVOKE_OTHER_CONTRACT_PROCEDURE(QX, IssueAsset, locals.input, locals.output, NOSTROMO_QX_TOKEN_ISSUANCE_FEE);

			if (locals.output.issuedNumberOfShares == state.projects.get(locals.tmpFundraising.indexOfProject).supplyOfToken)
			{
				locals.tmpFundraising.isCreatedToken = 1;

				locals.TransferShareManagementRightsInput.asset.assetName = state.projects.get(locals.tmpFundraising.indexOfProject).tokenName;
				locals.TransferShareManagementRightsInput.asset.issuer = SELF;
				locals.TransferShareManagementRightsInput.newManagingContractIndex = SELF_INDEX;
				locals.TransferShareManagementRightsInput.numberOfShares = state.projects.get(locals.tmpFundraising.indexOfProject).supplyOfToken;

				INVOKE_OTHER_CONTRACT_PROCEDURE(QX, TransferShareManagementRights, locals.TransferShareManagementRightsInput, locals.TransferShareManagementRightsOutput, 0);

				qpi.transferShareOwnershipAndPossession(state.projects.get(locals.tmpFundraising.indexOfProject).tokenName, SELF, SELF, SELF, state.projects.get(locals.tmpFundraising.indexOfProject).supplyOfToken - locals.tmpFundraising.soldAmount, state.projects.get(locals.tmpFundraising.indexOfProject).creator);
			}
		}

		state.fundaraisings.set(input.indexOfFundraising, locals.tmpFundraising);
		
	}

	struct claimToken_locals
	{
		investInfo tmpInvestData;
		uint64 maxClaimAmount, investedAmount, dayA, dayB, start_cur_diffSecond, cur_end_diffSecond, claimedAmount;
		uint32 curDate, tmpDate, numberOfInvestedProjects;
		sint32 i, j;
		uint8 curVestingStep, vestingPercent;
	};

	PUBLIC_PROCEDURE_WITH_LOCALS(claimToken)
	{
		packQuotteryDate(qpi.year(), qpi.month(), qpi.day(), qpi.hour(), qpi.minute(), qpi.second(), locals.curDate);

		if (input.indexOfFundraising >= state.numberOfFundraising)
		{
			return ;
		}

		state.investors.get(qpi.invocator(), state.tmpInvestedList);
		if (state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects) == 0)
		{
			return ;
		}

		for (locals.i = 0; locals.i < (sint32)locals.numberOfInvestedProjects; locals.i++)
		{
			if (state.tmpInvestedList.get(locals.i).indexOfFundraising == input.indexOfFundraising)
			{
				locals.investedAmount = state.tmpInvestedList.get(locals.i).investedAmount;
				locals.claimedAmount = state.tmpInvestedList.get(locals.i).claimedAmount;
				locals.tmpInvestData = state.tmpInvestedList.get(locals.i);
				break;
			}
		}

		if (locals.i == locals.numberOfInvestedProjects)
		{
			return ;
		}

		if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).listingStartDate && locals.curDate < state.fundaraisings.get(input.indexOfFundraising).cliffEndDate)
		{
			locals.maxClaimAmount = div(div(locals.investedAmount, state.fundaraisings.get(input.indexOfFundraising).tokenPrice) * state.fundaraisings.get(input.indexOfFundraising).TGE, 100ULL);
		}
		else if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).cliffEndDate && locals.curDate < state.fundaraisings.get(input.indexOfFundraising).vestingEndDate)
		{
			locals.tmpDate = state.fundaraisings.get(input.indexOfFundraising).cliffEndDate;
			diffDateInSecond(locals.tmpDate, locals.curDate, locals.j, locals.dayA, locals.dayB, locals.start_cur_diffSecond);
			locals.tmpDate = state.fundaraisings.get(input.indexOfFundraising).vestingEndDate;
			diffDateInSecond(locals.curDate, locals.tmpDate, locals.j, locals.dayA, locals.dayB, locals.cur_end_diffSecond);

			locals.curVestingStep = (uint8)div(locals.start_cur_diffSecond, div(locals.start_cur_diffSecond + locals.cur_end_diffSecond, state.fundaraisings.get(input.indexOfFundraising).stepOfVesting * 1ULL)) + 1;
			locals.vestingPercent = (uint8)div(100ULL - state.fundaraisings.get(input.indexOfFundraising).TGE, state.fundaraisings.get(input.indexOfFundraising).stepOfVesting * 1ULL) * locals.curVestingStep;
			locals.maxClaimAmount = div(div(locals.investedAmount, state.fundaraisings.get(input.indexOfFundraising).tokenPrice) * (state.fundaraisings.get(input.indexOfFundraising).TGE + locals.vestingPercent), 100ULL);
		}
		else if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).vestingEndDate)
		{
			locals.maxClaimAmount = div(locals.investedAmount, state.fundaraisings.get(input.indexOfFundraising).tokenPrice);
		}

		if (input.amount + locals.claimedAmount > locals.maxClaimAmount)
		{
			return ;
		}
		else 
		{
			qpi.transferShareOwnershipAndPossession(state.projects.get(state.fundaraisings.get(input.indexOfFundraising).indexOfProject).tokenName, SELF, SELF, SELF, input.amount, qpi.invocator());
			if (input.amount + locals.claimedAmount == locals.maxClaimAmount && state.fundaraisings.get(input.indexOfFundraising).vestingEndDate <= locals.curDate)
			{
				state.tmpInvestedList.set(locals.i, state.tmpInvestedList.get(locals.numberOfInvestedProjects - 1));
				state.numberOfInvestedProjects.set(qpi.invocator(), locals.numberOfInvestedProjects - 1);
			}
			else 
			{
				locals.tmpInvestData.claimedAmount = input.amount + locals.claimedAmount;
				state.tmpInvestedList.set(locals.i, locals.tmpInvestData);
			}
			state.investors.set(qpi.invocator(), state.tmpInvestedList);
			state.numberOfInvestedProjects.get(qpi.invocator(), locals.numberOfInvestedProjects);
			if (locals.numberOfInvestedProjects == 0)
			{
				state.investors.removeByKey(qpi.invocator());
				state.numberOfInvestedProjects.removeByKey(qpi.invocator());
			}
			output.claimedAmount = input.amount;
		}
	}

	struct upgradeTier_locals
	{
		uint64 deltaAmount;
		uint32 i, deltaPoolWeight;
		uint8 currentTierLevel;
	};

	PUBLIC_PROCEDURE_WITH_LOCALS(upgradeTier)
	{
		if (state.users.contains(qpi.invocator()) == 0)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}

		state.users.get(qpi.invocator(), locals.currentTierLevel);

		switch (locals.currentTierLevel)
		{
			case 1:
				locals.deltaAmount = NOSTROMO_TIER_CHESTBURST_STAKE_AMOUNT - NOSTROMO_TIER_FACEHUGGER_STAKE_AMOUNT;
				locals.deltaPoolWeight = NOSTROMO_TIER_CHESTBURST_POOL_WEIGHT - NOSTROMO_TIER_FACEHUGGER_POOL_WEIGHT;
				break;
			case 2:
				locals.deltaAmount = NOSTROMO_TIER_DOG_STAKE_AMOUNT - NOSTROMO_TIER_CHESTBURST_STAKE_AMOUNT;
				locals.deltaPoolWeight = NOSTROMO_TIER_DOG_POOL_WEIGHT - NOSTROMO_TIER_CHESTBURST_POOL_WEIGHT;
				break;
			case 3:
				locals.deltaAmount = NOSTROMO_TIER_XENOMORPH_STAKE_AMOUNT - NOSTROMO_TIER_DOG_STAKE_AMOUNT;
				locals.deltaPoolWeight = NOSTROMO_TIER_XENOMORPH_POOL_WEIGHT - NOSTROMO_TIER_DOG_POOL_WEIGHT;
				break;
			case 4:
				locals.deltaAmount = NOSTROMO_TIER_WARRIOR_STAKE_AMOUNT - NOSTROMO_TIER_XENOMORPH_STAKE_AMOUNT;
				locals.deltaPoolWeight = NOSTROMO_TIER_WARRIOR_POOL_WEIGHT - NOSTROMO_TIER_XENOMORPH_POOL_WEIGHT;
				break;
			default:
				break;
		}
		if (input.newTierLevel != locals.currentTierLevel + 1 || qpi.invocationReward() < (sint64)locals.deltaAmount)
		{
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
			return ;
		}
		else 
		{
			state.users.set(qpi.invocator(), input.newTierLevel);
			if (qpi.invocationReward() > (sint64)locals.deltaAmount)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward() - locals.deltaAmount);
			}
			state.totalPoolWeight += locals.deltaPoolWeight;
		}
	}

	PUBLIC_PROCEDURE(TransferShareManagementRights)
	{
		if (qpi.invocationReward() < state.transferRightsFee)
		{
			return ;
		}

		if (qpi.numberOfPossessedShares(input.asset.assetName, input.asset.issuer,qpi.invocator(), qpi.invocator(), SELF_INDEX, SELF_INDEX) < input.numberOfShares)
		{
			// not enough shares available
			output.transferredNumberOfShares = 0;
			if (qpi.invocationReward() > 0)
			{
				qpi.transfer(qpi.invocator(), qpi.invocationReward());
			}
		}
		else
		{
			if (qpi.releaseShares(input.asset, qpi.invocator(), qpi.invocator(), input.numberOfShares,
				input.newManagingContractIndex, input.newManagingContractIndex, state.transferRightsFee) < 0)
			{
				// error
				output.transferredNumberOfShares = 0;
				if (qpi.invocationReward() > 0)
				{
					qpi.transfer(qpi.invocator(), qpi.invocationReward());
				}
			}
			else
			{
				// success
				output.transferredNumberOfShares = input.numberOfShares;
				qpi.transfer(id(QX_CONTRACT_INDEX, 0, 0, 0), state.transferRightsFee);
				if (qpi.invocationReward() > state.transferRightsFee)
				{
					qpi.transfer(qpi.invocator(), qpi.invocationReward() -  state.transferRightsFee);
				}
			}
		}
	}

	PUBLIC_FUNCTION(getStats)
	{
		output.epochRevenue = state.epochRevenue;
		output.numberOfCreatedProject = state.numberOfCreatedProject;
		output.numberOfFundraising = state.numberOfFundraising;
		output.numberOfRegister = state.numberOfRegister;
		output.totalPoolWeight = state.totalPoolWeight;
	}

	PUBLIC_FUNCTION(getTierLevelByUser)
	{
		state.users.get(input.userId, output.tierLevel);
	}

	PUBLIC_FUNCTION(getUserVoteStatus)
	{
		state.numberOfVotedProject.get(input.userId, output.numberOfVotedProjects);
		state.voteStatus.get(input.userId, output.projectIndexList);
	}

	PUBLIC_FUNCTION(checkTokenCreatability)
	{
		output.result = state.tokens.contains(input.tokenName);
	}

	PUBLIC_FUNCTION(getNumberOfInvestedProjects)
	{
		state.numberOfInvestedProjects.get(input.userId, output.numberOfInvestedProjects);
	}

public:
	struct getProjectByIndex_input
	{
		uint32 indexOfProject;
	};

	struct getProjectByIndex_output
	{
		projectInfo project;
	};

	PUBLIC_FUNCTION(getProjectByIndex)
	{
		output.project = state.projects.get(input.indexOfProject);
	}

	struct getFundarasingByIndex_input
	{
		uint32 indexOfFundarasing;
	};

	struct getFundarasingByIndex_output
	{
		fundaraisingInfo fundarasing;
	};

	PUBLIC_FUNCTION(getFundarasingByIndex)
	{
		output.fundarasing = state.fundaraisings.get(input.indexOfFundarasing);
	}

	struct getProjectIndexListByCreator_input
	{
		id creator;
	};

	struct getProjectIndexListByCreator_output
	{
		Array<uint32, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST> indexListForProjects;
	};

	struct getProjectIndexListByCreator_locals
	{
		uint32 i, countOfProject;
	};

	PUBLIC_FUNCTION_WITH_LOCALS(getProjectIndexListByCreator)
	{
		for (locals.i = 0; locals.i < state.numberOfCreatedProject; locals.i++)
		{
			if (state.projects.get(locals.i).creator == input.creator)
			{
				output.indexListForProjects.set(locals.countOfProject++, locals.i);
			}
		}
		for (locals.i = locals.countOfProject; locals.i < NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST; locals.i++)
		{
			output.indexListForProjects.set(locals.i, NOSTROMO_MAX_NUMBER_PROJECT);
		}
	}

	struct getInfoUserInvested_input
	{
		id investorId;
	};

	struct getInfoUserInvested_output
	{
		Array<investInfo, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST> listUserInvested;
	};

	struct getInfoUserInvested_locals
	{
		uint32 i, countOfProject;
	};

	PUBLIC_FUNCTION_WITH_LOCALS(getInfoUserInvested)
	{
		state.investors.get(input.investorId, output.listUserInvested);
	}

	struct getMaxClaimAmount_input
	{
		id investorId;
		uint32 indexOfFundraising;
	};

	struct getMaxClaimAmount_output
	{
		uint64 amount;
	};

	struct getMaxClaimAmount_locals
	{
		Array<investInfo, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST> tmpInvestedList;
		investInfo tmpInvestData;
		uint64 maxClaimAmount, investedAmount, dayA, dayB, dayC, dayD, start_cur_diffSecond, cur_end_diffSecond, claimedAmount;
		uint32 curDate, tmpDate, numberOfInvestedProjects;
		sint32 i, j, k;
		uint8 curVestingStep, vestingPercent;
		bit flag;
	};

	PUBLIC_FUNCTION_WITH_LOCALS(getMaxClaimAmount)
	{
		packQuotteryDate(qpi.year(), qpi.month(), qpi.day(), qpi.hour(), qpi.minute(), qpi.second(), locals.curDate);

		if (input.indexOfFundraising >= state.numberOfFundraising)
		{
			return ;
		}

		state.investors.get(input.investorId, locals.tmpInvestedList);
		if (state.numberOfInvestedProjects.get(input.investorId, locals.numberOfInvestedProjects) == 0)
		{
			return ;
		}

		for (locals.i = 0; locals.i < (sint32)locals.numberOfInvestedProjects; locals.i++)
		{
			if (locals.tmpInvestedList.get(locals.i).indexOfFundraising == input.indexOfFundraising)
			{
				locals.investedAmount = locals.tmpInvestedList.get(locals.i).investedAmount;
				locals.claimedAmount = locals.tmpInvestedList.get(locals.i).claimedAmount;
				locals.tmpInvestData = locals.tmpInvestedList.get(locals.i);
				break;
			}
		}

		if (locals.i == locals.numberOfInvestedProjects)
		{
			return ;
		}

		if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).listingStartDate && locals.curDate < state.fundaraisings.get(input.indexOfFundraising).cliffEndDate)
		{
			output.amount = div(div(locals.investedAmount, state.fundaraisings.get(input.indexOfFundraising).tokenPrice) * state.fundaraisings.get(input.indexOfFundraising).TGE, 100ULL);
		}
		else if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).cliffEndDate && locals.curDate < state.fundaraisings.get(input.indexOfFundraising).vestingEndDate)
		{
			locals.tmpDate = state.fundaraisings.get(input.indexOfFundraising).cliffEndDate;
			diffDateInSecond(locals.tmpDate, locals.curDate, locals.j, locals.dayA, locals.dayB, locals.start_cur_diffSecond);
			locals.tmpDate = state.fundaraisings.get(input.indexOfFundraising).vestingEndDate;
			diffDateInSecond(locals.curDate, locals.tmpDate, locals.k, locals.dayC, locals.dayD, locals.cur_end_diffSecond);

			locals.curVestingStep = (uint8)div(locals.start_cur_diffSecond, div(locals.start_cur_diffSecond + locals.cur_end_diffSecond, state.fundaraisings.get(input.indexOfFundraising).stepOfVesting * 1ULL)) + 1;
			locals.vestingPercent = (uint8)div(100ULL - state.fundaraisings.get(input.indexOfFundraising).TGE, state.fundaraisings.get(input.indexOfFundraising).stepOfVesting * 1ULL) * locals.curVestingStep;
			output.amount = div(div(locals.investedAmount, state.fundaraisings.get(input.indexOfFundraising).tokenPrice) * (state.fundaraisings.get(input.indexOfFundraising).TGE + locals.vestingPercent), 100ULL);
		}
		else if (locals.curDate >= state.fundaraisings.get(input.indexOfFundraising).vestingEndDate)
		{
			output.amount = div(locals.investedAmount, state.fundaraisings.get(input.indexOfFundraising).tokenPrice);
		}
	}

    REGISTER_USER_FUNCTIONS_AND_PROCEDURES()
	{
		REGISTER_USER_FUNCTION(getStats, 1);
		REGISTER_USER_FUNCTION(getTierLevelByUser, 2);
		REGISTER_USER_FUNCTION(getUserVoteStatus, 3);
		REGISTER_USER_FUNCTION(checkTokenCreatability, 4);
		REGISTER_USER_FUNCTION(getNumberOfInvestedProjects, 5);
		REGISTER_USER_FUNCTION(getProjectByIndex, 6);
		REGISTER_USER_FUNCTION(getFundarasingByIndex, 7);
		REGISTER_USER_FUNCTION(getProjectIndexListByCreator, 8);
		REGISTER_USER_FUNCTION(getInfoUserInvested, 9);
		REGISTER_USER_FUNCTION(getMaxClaimAmount, 10);
		
		REGISTER_USER_PROCEDURE(registerInTier, 1);
		REGISTER_USER_PROCEDURE(logoutFromTier, 2);
		REGISTER_USER_PROCEDURE(createProject, 3);
		REGISTER_USER_PROCEDURE(voteInProject, 4);
		REGISTER_USER_PROCEDURE(createFundraising, 5);
		REGISTER_USER_PROCEDURE(investInProject, 6);
		REGISTER_USER_PROCEDURE(claimToken, 7);
		REGISTER_USER_PROCEDURE(upgradeTier, 8);
		REGISTER_USER_PROCEDURE(TransferShareManagementRights, 9);
	}

	INITIALIZE()
	{
		state.teamAddress = ID(_G, _E, _H, _N, _R, _F, _U, _O, _I, _I, _C, _S, _B, _C, _S, _R, _F, _M, _N, _J, _T, _C, _J, _K, _C, _J, _H, _A, _T, _Z, _X, _A, _X, _Y, _O, _F, _W, _X, _U, _F, _L, _C, _K, _F, _P, _B, _W, _X, _Q, _A, _C, _B, _S, _Z, _F, _F);
		state.transferRightsFee = 100;
	}

	struct END_EPOCH_locals
	{
		fundaraisingInfo tmpFundraising;
		investInfo tmpInvest;
		Array<uint32, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST> votedList;
		Array<uint32, NOSTROMO_MAX_NUMBER_OF_PROJECT_USER_INVEST> clearedVotedList;
		id userId;
		sint64 idx;
		uint32 numberOfVotedProject, clearedNumberOfVotedProject, i, j, curDate, indexOfProject, numberOfInvestedProjects, tierLevel;
	};

	END_EPOCH_WITH_LOCALS()
	{
		packQuotteryDate(qpi.year(), qpi.month(), qpi.day(), qpi.hour(), qpi.minute(), qpi.second(), locals.curDate);

		locals.idx = state.investors.nextElementIndex(NULL_INDEX);
		while (locals.idx != NULL_INDEX)
		{
			locals.userId = state.investors.key(locals.idx);
			state.investors.get(locals.userId, state.tmpInvestedList);
			state.numberOfInvestedProjects.get(locals.userId, locals.numberOfInvestedProjects);

			for (locals.i = 0; locals.i < locals.numberOfInvestedProjects; locals.i++)
			{
				if (state.fundaraisings.get(locals.i).thirdPhaseEndDate < locals.curDate && state.fundaraisings.get(locals.i).isCreatedToken == 0 && state.fundaraisings.get(locals.i).raisedFunds != 0)
				{	
					qpi.transfer(locals.userId, state.tmpInvestedList.get(locals.i).investedAmount);
					state.tmpInvestedList.set(locals.i, state.tmpInvestedList.get(--locals.numberOfInvestedProjects));
				}
			}
			if (locals.numberOfInvestedProjects == 0)
			{
				state.investors.removeByKey(locals.userId);
				state.numberOfInvestedProjects.removeByKey(locals.userId);
			}
			else
			{
				state.investors.set(locals.userId, state.tmpInvestedList);
				state.numberOfInvestedProjects.set(locals.userId, locals.numberOfInvestedProjects);
			}
			locals.idx = state.investors.nextElementIndex(locals.idx);
		}

		for (locals.i = 0; locals.i < state.numberOfFundraising; locals.i++)
		{
			if (state.fundaraisings.get(locals.i).thirdPhaseEndDate < locals.curDate && state.fundaraisings.get(locals.i).isCreatedToken == 0 && state.fundaraisings.get(locals.i).raisedFunds != 0)
			{
				locals.tmpFundraising = state.fundaraisings.get(locals.i);
				locals.tmpFundraising.raisedFunds = 0;
				state.fundaraisings.set(locals.i, locals.tmpFundraising);
			}
			else if (state.fundaraisings.get(locals.i).thirdPhaseEndDate < locals.curDate && state.fundaraisings.get(locals.i).isCreatedToken == 1 && state.fundaraisings.get(locals.i).raisedFunds != 0)
			{
				locals.tmpFundraising = state.fundaraisings.get(locals.i);

				state.epochRevenue += div(locals.tmpFundraising.raisedFunds * 5, 100ULL);
				qpi.transfer(state.projects.get(locals.tmpFundraising.indexOfProject).creator, locals.tmpFundraising.raisedFunds - div(locals.tmpFundraising.raisedFunds * 5, 100ULL));
				
				qpi.transferShareOwnershipAndPossession(state.projects.get(locals.tmpFundraising.indexOfProject).tokenName, SELF, SELF, SELF, state.fundaraisings.get(locals.i).soldAmount - div(locals.tmpFundraising.raisedFunds, state.fundaraisings.get(locals.i).tokenPrice), state.projects.get(locals.tmpFundraising.indexOfProject).creator);
				
				locals.tmpFundraising.raisedFunds = 0;
				state.fundaraisings.set(locals.i, locals.tmpFundraising);
			}
		}

		qpi.transfer(state.teamAddress, div(state.epochRevenue, 10ULL));
		state.epochRevenue -= div(state.epochRevenue, 10ULL);
		qpi.distributeDividends(div(state.epochRevenue, 676ULL));
		state.epochRevenue -= div(state.epochRevenue, 676ULL) * 676;
		
		locals.idx = state.users.nextElementIndex(NULL_INDEX);
		while (locals.idx != NULL_INDEX)
		{
			locals.userId = state.users.key(locals.idx);
			locals.tierLevel = state.users.value(locals.idx);

			if (state.numberOfVotedProject.get(locals.userId, locals.numberOfVotedProject))
			{
				state.voteStatus.get(locals.userId, locals.votedList);
				locals.clearedNumberOfVotedProject = 0;
				for (locals.j = 0; locals.j < locals.numberOfVotedProject; locals.j++)
				{
					locals.indexOfProject = locals.votedList.get(locals.j);

					if (state.projects.get(locals.indexOfProject).endDate > locals.curDate)
					{
						locals.clearedVotedList.set(locals.clearedNumberOfVotedProject++, locals.indexOfProject);
					}
				}
				if (locals.clearedNumberOfVotedProject == 0)
				{
					state.numberOfVotedProject.removeByKey(locals.userId);
					state.voteStatus.removeByKey(locals.userId);
				}
				else
				{
					state.numberOfVotedProject.set(locals.userId, locals.clearedNumberOfVotedProject);
					state.voteStatus.set(locals.userId, locals.clearedVotedList);
				}
			}

			locals.idx = state.users.nextElementIndex(locals.idx);
		}

		state.users.cleanupIfNeeded();
		state.investors.cleanupIfNeeded();
		state.numberOfInvestedProjects.cleanupIfNeeded();
		state.numberOfVotedProject.cleanupIfNeeded();
		state.voteStatus.cleanupIfNeeded();
	}

	PRE_ACQUIRE_SHARES()
    {
		output.allowTransfer = true;
    }
};