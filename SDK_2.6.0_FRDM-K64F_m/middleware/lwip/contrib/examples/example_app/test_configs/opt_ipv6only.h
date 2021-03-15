/* test an lwipopts.h file with default contents */
#define NO_SYS                          0
#define NO_SYS_NO_TIMERS                0
#define LWIP_TIMERS                     1
#define LWIP_TIMERS_CUSTOM              0
#define LWIP_MPU_COMPATIBLE             0
#define LWIP_TCPIP_CORE_LOCKING         1
#define LWIP_TCPIP_CORE_LOCKING_INPUT   0
#define SYS_LIGHTWEIGHT_PROT            1
#define MEM_LIBC_MALLOC                 0
#define MEMP_MEM_MALLOC                 0
#define MEMP_MEM_INIT                   0
#define MEM_ALIGNMENT                   1
#define MEM_SIZE                        1600
#define MEMP_OVERFLOW_CHECK             0
#define MEMP_SANITY_CHECK               0
#define MEM_OVERFLOW_CHECK              0
#define MEM_SANITY_CHECK                0
#define MEM_USE_POOLS                   0
#define MEM_USE_POOLS_TRY_BIGGER_POOL   0
#define MEMP_USE_CUSTOM_POOLS           0
#define LWIP_ALLOW_MEM_FREE_FROM_OTHER_CONTEXT 0
/*#define MEMP_NUM_PBUF                   16
#define MEMP_NUM_RAW_PCB                4
#define MEMP_NUM_UDP_PCB                4
#define MEMP_NUM_TCP_PCB                5
#define MEMP_NUM_TCP_PCB_LISTEN         8
#define MEMP_NUM_TCP_SEG                16
#define MEMP_NUM_ALTCP_PCB              MEMP_NUM_TCP_PCB
#define MEMP_NUM_REASSDATA              5
#define MEMP_NUM_FRAG_PBUF              15
#define MEMP_NUM_ARP_QUEUE              30
#define MEMP_NUM_IGMP_GROUP             8
#define MEMP_NUM_SYS_TIMEOUT            (LWIP_NUM_SYS_TIMEOUT_INTERNAL + 2)
#define MEMP_NUM_NETBUF                 2
#define MEMP_NUM_NETCONN                4
#define MEMP_NUM_SELECT_CB              4
#define MEMP_NUM_TCPIP_MSG_API          8
#define MEMP_NUM_TCPIP_MSG_INPKT        8
#define MEMP_NUM_NETDB                  1
#define MEMP_NUM_LOCALHOSTLIST          1
#define PBUF_POOL_SIZE                  16
#define MEMP_NUM_API_MSG                MEMP_NUM_TCPIP_MSG_API
#define MEMP_NUM_DNS_API_MSG            MEMP_NUM_TCPIP_MSG_API
#define MEMP_NUM_SOCKET_SETGETSOCKOPT_DATA MEMP_NUM_TCPIP_MSG_API
#define MEMP_NUM_NETIFAPI_MSG           MEMP_NUM_TCPIP_MSG_API*/
#define LWIP_ARP                        1
#define ARP_TABLE_SIZE                  10
#define ARP_MAXAGE                      300
#define ARP_QUEUEING                    0
#define ARP_QUEUE_LEN                   3
#define ETHARP_SUPPORT_VLAN             0
#define LWIP_ETHERNET                   LWIP_ARP
#define ETH_PAD_SIZE                    0
#define ETHARP_SUPPORT_STATIC_ENTRIES   0
#define ETHARP_TABLE_MATCH_NETIF        !LWIP_SINGLE_NETIF
#define LWIP_IPV4                       0
#define IP_FORWARD                      0
#define IP_REASSEMBLY                   1
#define IP_FRAG                         1
#define IP_OPTIONS_ALLOWED              1
#define IP_REASS_MAXAGE                 15
#define IP_REASS_MAX_PBUFS              10
#define IP_DEFAULT_TTL                  255
#define IP_SOF_BROADCAST                0
#define IP_SOF_BROADCAST_RECV           0
#define IP_FORWARD_ALLOW_TX_ON_RX_NETIF 0
#define LWIP_ICMP                       1
#define ICMP_TTL                        (IP_DEFAULT_TTL)
#define LWIP_BROADCAST_PING             0
#define LWIP_MULTICAST_PING             0
#define LWIP_RAW                        0
#define RAW_TTL                         (IP_DEFAULT_TTL)
#define LWIP_DHCP                       0
#define DHCP_DOES_ARP_CHECK             ((LWIP_DHCP) && (LWIP_ARP))
#define LWIP_DHCP_CHECK_LINK_UP         0
#define LWIP_DHCP_BOOTP_FILE            0
#define LWIP_DHCP_GET_NTP_SRV           0
#define LWIP_DHCP_MAX_NTP_SERVERS       1
#define LWIP_DHCP_MAX_DNS_SERVERS       DNS_MAX_SERVERS
#define LWIP_AUTOIP                     0
#define LWIP_DHCP_AUTOIP_COOP           0
#define LWIP_DHCP_AUTOIP_COOP_TRIES     9
#define LWIP_MIB2_CALLBACKS             0
#define LWIP_MULTICAST_TX_OPTIONS       ((LWIP_IGMP || LWIP_IPV6_MLD) && (LWIP_UDP || LWIP_RAW))
#define LWIP_IGMP                       0
#define LWIP_DNS                        0
#define DNS_TABLE_SIZE                  4
#define DNS_MAX_NAME_LENGTH             256
#define DNS_MAX_SERVERS                 2
#define DNS_MAX_RETRIES                 4
#define DNS_DOES_NAME_CHECK             1
#define LWIP_DNS_SECURE (LWIP_DNS_SECURE_RAND_XID | LWIP_DNS_SECURE_NO_MULTIPLE_OUTSTANDING | LWIP_DNS_SECURE_RAND_SRC_PORT)
#define DNS_LOCAL_HOSTLIST              0
#define DNS_LOCAL_HOSTLIST_IS_DYNAMIC   0
#define LWIP_DNS_SUPPORT_MDNS_QUERIES   0
#define LWIP_UDP                        1
#define LWIP_UDPLITE                    0
#define UDP_TTL                         (IP_DEFAULT_TTL)
#define LWIP_NETBUF_RECVINFO            0
#define LWIP_TCP                        1
#define TCP_TTL                         (IP_DEFAULT_TTL)
#define TCP_WND                         (4 * TCP_MSS)
#define TCP_MAXRTX                      12
#define TCP_SYNMAXRTX                   6
#define TCP_QUEUE_OOSEQ                 (LWIP_TCP)
#define LWIP_TCP_SACK_OUT               0
#define LWIP_TCP_MAX_SACK_NUM           4
#define TCP_MSS                         536
#define TCP_CALCULATE_EFF_SEND_MSS      1
#define TCP_SND_BUF                     (2 * TCP_MSS)
#define TCP_SND_QUEUELEN                ((4 * (TCP_SND_BUF) + (TCP_MSS - 1))/(TCP_MSS))
#define TCP_SNDLOWAT                    LWIP_MIN(LWIP_MAX(((TCP_SND_BUF)/2), (2 * TCP_MSS) + 1), (TCP_SND_BUF) - 1)
#define TCP_SNDQUEUELOWAT               LWIP_MAX(((TCP_SND_QUEUELEN)/2), 5)
#define TCP_OOSEQ_MAX_BYTES             0
#define TCP_OOSEQ_BYTES_LIMIT(pcb)      TCP_OOSEQ_MAX_BYTES
#define TCP_OOSEQ_MAX_PBUFS             0
#define TCP_OOSEQ_PBUFS_LIMIT(pcb)      TCP_OOSEQ_MAX_PBUFS
#define TCP_LISTEN_BACKLOG              0
#define TCP_DEFAULT_LISTEN_BACKLOG      0xff
#define TCP_OVERSIZE                    TCP_MSS
#define LWIP_TCP_TIMESTAMPS             0
#define TCP_WND_UPDATE_THRESHOLD        LWIP_MIN((TCP_WND / 4), (TCP_MSS * 4))
#define LWIP_EVENT_API                  0
#define LWIP_CALLBACK_API               1
#define LWIP_WND_SCALE                  0
#define TCP_RCV_SCALE                   0
#define LWIP_TCP_PCB_NUM_EXT_ARGS       0
#define LWIP_ALTCP                      0
#define LWIP_ALTCP_TLS                  0
#define PBUF_LINK_HLEN                  (14 + ETH_PAD_SIZE)
#define PBUF_LINK_ENCAPSULATION_HLEN    0
#define PBUF_POOL_BUFSIZE               LWIP_MEM_ALIGN_SIZE(TCP_MSS+40+PBUF_LINK_ENCAPSULATION_HLEN+PBUF_LINK_HLEN)
#define LWIP_PBUF_REF_T                 u8_t
#define LWIP_SINGLE_NETIF               0
#define LWIP_NETIF_HOSTNAME             0
#define LWIP_NETIF_API                  0
#define LWIP_NETIF_STATUS_CALLBACK      0
#define LWIP_NETIF_EXT_STATUS_CALLBACK  0
#define LWIP_NETIF_LINK_CALLBACK        0
#define LWIP_NETIF_REMOVE_CALLBACK      0
#define LWIP_NETIF_HWADDRHINT           0
#define LWIP_NETIF_TX_SINGLE_PBUF       0
#define LWIP_NUM_NETIF_CLIENT_DATA      0
#define LWIP_HAVE_LOOPIF                (LWIP_NETIF_LOOPBACK && !LWIP_SINGLE_NETIF)
#define LWIP_LOOPIF_MULTICAST           0
#define LWIP_NETIF_LOOPBACK             0
#define LWIP_LOOPBACK_MAX_PBUFS         0
#define LWIP_NETIF_LOOPBACK_MULTITHREADING    (!NO_SYS)
/*#define TCPIP_THREAD_NAME               "tcpip_thread"
#define TCPIP_THREAD_STACKSIZE          0
#define TCPIP_THREAD_PRIO               1
#define TCPIP_MBOX_SIZE                 0
#define LWIP_TCPIP_THREAD_ALIVE()
#define SLIPIF_THREAD_NAME              "slipif_loop"
#define SLIPIF_THREAD_STACKSIZE         0
#define SLIPIF_THREAD_PRIO              1
#define DEFAULT_THREAD_NAME             "lwIP"
#define DEFAULT_THREAD_STACKSIZE        0
#define DEFAULT_THREAD_PRIO             1
#define DEFAULT_RAW_RECVMBOX_SIZE       0
#define DEFAULT_UDP_RECVMBOX_SIZE       0
#define DEFAULT_TCP_RECVMBOX_SIZE       0
#define DEFAULT_ACCEPTMBOX_SIZE         0*/
#define LWIP_NETCONN                    1
#define LWIP_TCPIP_TIMEOUT              0
#define LWIP_NETCONN_SEM_PER_THREAD     0
#define LWIP_NETCONN_FULLDUPLEX         0
#define LWIP_SOCKET                     1
#define LWIP_COMPAT_SOCKETS             1 /* 0..2 */
#define LWIP_POSIX_SOCKETS_IO_NAMES     1
#define LWIP_SOCKET_OFFSET              0
#define LWIP_TCP_KEEPALIVE              0
#define LWIP_SO_SNDTIMEO                0
#define LWIP_SO_RCVTIMEO                0
#define LWIP_SO_SNDRCVTIMEO_NONSTANDARD 0
#define LWIP_SO_RCVBUF                  0
#define LWIP_SO_LINGER                  0
#define RECV_BUFSIZE_DEFAULT            INT_MAX
#define LWIP_TCP_CLOSE_TIMEOUT_MS_DEFAULT 20000
#define SO_REUSE                        0
#define SO_REUSE_RXTOALL                0
#define LWIP_FIONREAD_LINUXMODE         0
#define LWIP_SOCKET_SELECT              1
#define LWIP_SOCKET_POLL                1
#define LWIP_STATS                      1
#define LWIP_STATS_DISPLAY              0
#define LINK_STATS                      1
#define ETHARP_STATS                    (LWIP_ARP)
#define IP_STATS                        1
#define IPFRAG_STATS                    (IP_REASSEMBLY || IP_FRAG)
#define ICMP_STATS                      1
#define IGMP_STATS                      (LWIP_IGMP)
#define UDP_STATS                       (LWIP_UDP)
#define TCP_STATS                       (LWIP_TCP)
#define MEM_STATS                       ((MEM_LIBC_MALLOC == 0) && (MEM_USE_POOLS == 0))
#define MEMP_STATS                      (MEMP_MEM_MALLOC == 0)
#define SYS_STATS                       (NO_SYS == 0)
#define IP6_STATS                       (LWIP_IPV6)
#define ICMP6_STATS                     (LWIP_IPV6 && LWIP_ICMP6)
#define IP6_FRAG_STATS                  (LWIP_IPV6 && (LWIP_IPV6_FRAG || LWIP_IPV6_REASS))
#define MLD6_STATS                      (LWIP_IPV6 && LWIP_IPV6_MLD)
#define ND6_STATS                       (LWIP_IPV6)
#define MIB2_STATS                      0
#define LWIP_CHECKSUM_CTRL_PER_NETIF    0
#define CHECKSUM_GEN_IP                 1
#define CHECKSUM_GEN_UDP                1
#define CHECKSUM_GEN_TCP                1
#define CHECKSUM_GEN_ICMP               1
#define CHECKSUM_GEN_ICMP6              1
#define CHECKSUM_CHECK_IP               1
#define CHECKSUM_CHECK_UDP              1
#define CHECKSUM_CHECK_TCP              1
#define CHECKSUM_CHECK_ICMP             1
#define CHECKSUM_CHECK_ICMP6            1
#define LWIP_CHECKSUM_ON_COPY           0
#define LWIP_IPV6                       1
#define IPV6_REASS_MAXAGE               60
#define LWIP_IPV6_SCOPES                (LWIP_IPV6 && !LWIP_SINGLE_NETIF)
#define LWIP_IPV6_SCOPES_DEBUG          0
#define LWIP_IPV6_NUM_ADDRESSES         3
#define LWIP_IPV6_FORWARD               0
#define LWIP_IPV6_FRAG                  1
#define LWIP_IPV6_REASS                 (LWIP_IPV6)
#define LWIP_IPV6_SEND_ROUTER_SOLICIT   1
#define LWIP_IPV6_AUTOCONFIG            (LWIP_IPV6)
#define LWIP_IPV6_ADDRESS_LIFETIMES     (LWIP_IPV6_AUTOCONFIG)
#define LWIP_IPV6_DUP_DETECT_ATTEMPTS   1
#define LWIP_ICMP6                      (LWIP_IPV6)
#define LWIP_ICMP6_DATASIZE             8
#define LWIP_ICMP6_HL                   255
#define LWIP_IPV6_MLD                   (LWIP_IPV6)
#define MEMP_NUM_MLD6_GROUP             4
#define LWIP_ND6_QUEUEING               (LWIP_IPV6)
#define MEMP_NUM_ND6_QUEUE              20
#define LWIP_ND6_NUM_NEIGHBORS          10
#define LWIP_ND6_NUM_DESTINATIONS       10
#define LWIP_ND6_NUM_PREFIXES           5
#define LWIP_ND6_NUM_ROUTERS            3
#define LWIP_ND6_MAX_MULTICAST_SOLICIT  3
#define LWIP_ND6_MAX_UNICAST_SOLICIT    3
#define LWIP_ND6_MAX_ANYCAST_DELAY_TIME 1000
#define LWIP_ND6_MAX_NEIGHBOR_ADVERTISEMENT  3
#define LWIP_ND6_REACHABLE_TIME         30000
#define LWIP_ND6_RETRANS_TIMER          1000
#define LWIP_ND6_DELAY_FIRST_PROBE_TIME 5000
#define LWIP_ND6_ALLOW_RA_UPDATES       1
#define LWIP_ND6_TCP_REACHABILITY_HINTS 1
#define LWIP_ND6_RDNSS_MAX_DNS_SERVERS  0
#define LWIP_IPV6_DHCP6                 0
#define LWIP_IPV6_DHCP6_STATEFUL        0
#define LWIP_IPV6_DHCP6_STATELESS       LWIP_IPV6_DHCP6
#define LWIP_DHCP6_GET_NTP_SRV          0
#define LWIP_DHCP6_MAX_NTP_SERVERS      1
#define LWIP_DHCP6_MAX_DNS_SERVERS      DNS_MAX_SERVERS

/* TODO: check hooks */

#define LWIP_DBG_MIN_LEVEL              LWIP_DBG_LEVEL_ALL
#define LWIP_DBG_TYPES_ON               LWIP_DBG_ON
#define ETHARP_DEBUG                    LWIP_DBG_OFF
#define NETIF_DEBUG                     LWIP_DBG_OFF
#define PBUF_DEBUG                      LWIP_DBG_OFF
#define API_LIB_DEBUG                   LWIP_DBG_OFF
#define API_MSG_DEBUG                   LWIP_DBG_OFF
#define SOCKETS_DEBUG                   LWIP_DBG_OFF
#define ICMP_DEBUG                      LWIP_DBG_OFF
#define IGMP_DEBUG                      LWIP_DBG_OFF
#define INET_DEBUG                      LWIP_DBG_OFF
#define IP_DEBUG                        LWIP_DBG_OFF
#define IP_REASS_DEBUG                  LWIP_DBG_OFF
#define RAW_DEBUG                       LWIP_DBG_OFF
#define MEM_DEBUG                       LWIP_DBG_OFF
#define MEMP_DEBUG                      LWIP_DBG_OFF
#define SYS_DEBUG                       LWIP_DBG_OFF
#define TIMERS_DEBUG                    LWIP_DBG_OFF
#define TCP_DEBUG                       LWIP_DBG_OFF
#define TCP_INPUT_DEBUG                 LWIP_DBG_OFF
#define TCP_FR_DEBUG                    LWIP_DBG_OFF
#define TCP_RTO_DEBUG                   LWIP_DBG_OFF
#define TCP_CWND_DEBUG                  LWIP_DBG_OFF
#define TCP_WND_DEBUG                   LWIP_DBG_OFF
#define TCP_OUTPUT_DEBUG                LWIP_DBG_OFF
#define TCP_RST_DEBUG                   LWIP_DBG_OFF
#define TCP_QLEN_DEBUG                  LWIP_DBG_OFF
#define UDP_DEBUG                       LWIP_DBG_OFF
#define TCPIP_DEBUG                     LWIP_DBG_OFF
#define SLIP_DEBUG                      LWIP_DBG_OFF
#define DHCP_DEBUG                      LWIP_DBG_OFF
#define AUTOIP_DEBUG                    LWIP_DBG_OFF
#define DNS_DEBUG                       LWIP_DBG_OFF
#define IP6_DEBUG                       LWIP_DBG_OFF
#define DHCP6_DEBUG                     LWIP_DBG_OFF
#define LWIP_TESTMODE                   0

#define LWIP_PERF                       0
