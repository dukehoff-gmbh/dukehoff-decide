/*
 * Copyright 2026 Dukehoff GmbH
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.dukehoff.decide.showcase;

import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;

/** The typed decision for one complaint of the CFPB narratives archive: which financial product it is about. */
@Question("You route complaints that consumers send to a financial regulator about banks and other financial companies.")
public record ConsumerComplaint(@Question("Which financial product is the complaint about?") Product product) {

    public enum Product {
        @Option("Credit reports and credit scores: wrong or unknown entries, disputes with credit bureaus, "
                + "identity theft on the report") CREDIT_REPORTING,
        @Option("Debt collectors and attempts to collect a debt") DEBT_COLLECTION,
        @Option("Credit card accounts, charges, fees and rewards") CREDIT_CARD,
        @Option("Prepaid cards and gift cards") PREPAID_CARD,
        @Option("Checking or savings accounts: deposits, withdrawals, overdrafts, closed accounts") CHECKING_OR_SAVINGS,
        @Option("Mortgages and home loans") MORTGAGE,
        @Option("Student loans") STUDENT_LOAN,
        @Option("Car loans and leases") VEHICLE_LOAN,
        @Option("Payday loans, title loans, personal loans, cash advances") PERSONAL_LOAN,
        @Option("Money transfers, payment apps, mobile wallets, virtual currency") MONEY_TRANSFER
    }
}
